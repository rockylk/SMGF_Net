"""Clean-only fixed-budget training, atomic resume, and paired evaluation."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
import baseline_v17_common as C
from ibgc_model import IBGC
from ibgc_io import (save_torch, load_checkpoint, write_json, save_npz, sha256,
                     read_json, bind_manifest)

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # PyG CUDA scatter may be nondeterministic. Warn rather than silently
    # claiming bitwise CUDA reproducibility; CPU resume is tested exactly.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def loader(dataset, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, generator=generator, drop_last=False)


@torch.no_grad()
def predict(model, dataset, device):
    model.eval()
    probs, labels, ids, losses = [], [], [], 0.0
    for b in loader(dataset):
        b = b.to(device)
        logp = model(b.x, b.edge_index, b.batch)
        if not torch.isfinite(logp).all():
            raise RuntimeError("Non-finite evaluation output")
        losses += F.nll_loss(logp, b.y, reduction="sum").item()
        probs.append(logp.exp().cpu().numpy())
        labels.append(b.y.cpu().numpy())
        ids.append(b.sample_id.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), np.concatenate(ids), losses / len(dataset)


def train(train_data, validation_data, num_classes, seed, device, out, identity,
          logger, resume=False, epochs=EPOCHS, stop_after=None):
    """Test-only stop_after simulates interruption; formal CLI never exposes it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bind_manifest(out / "identity.json", identity)
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    model = IBGC(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    last = out / "latest.pt"
    first, best, best_epoch, history, elapsed = 1, float("inf"), None, [], 0.0
    if last.exists():
        if not resume:
            raise RuntimeError("Existing checkpoint: use --resume or a new run directory")
        saved = load_checkpoint(last, identity)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        first = saved["epoch"] + 1
        best, best_epoch = saved["best_validation_ce"], saved["best_epoch"]
        history, elapsed = saved["history"], saved["elapsed_seconds"]
        # Latest is the authoritative transaction: repairs a best.pt written
        # just before an interruption that happened before latest.pt committed.
        save_torch(out / "best_validation.pt", dict(schema_version=1, identity=identity,
                   epoch=best_epoch, validation_ce=best,
                   model_state_dict=saved["best_model_state_dict"]))
        best_state = saved["best_model_state_dict"]
        restore_rng(saved["rng"], gen)
    else:
        if (out / "best_validation.pt").exists() and not resume:
            raise RuntimeError("Partial checkpoint found: use --resume")
        best_state = None
    if first > epochs + 1:
        raise RuntimeError("Checkpoint exceeds fixed epoch budget")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = loader(train_data, True, gen)
    for epoch in range(first, epochs + 1):
        tick = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_no, b in enumerate(train_loader, 1):
            b = b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logp = model(b.x, b.edge_index, b.batch)
            loss = F.nll_loss(logp, b.y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite train loss at epoch {}".format(epoch))
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            loss_sum += loss.item() * b.num_graphs
            seen += b.num_graphs
            if batch_no % 100 == 0:
                logger.log("[train][seed={}] epoch={}/{} batch={}/{} loss={:.5f}".format(
                    seed, epoch, epochs, batch_no, len(train_loader), loss_sum / seen))
        y, p, _, val_ce = predict(model, validation_data, device)
        if val_ce < best:  # strict improvement; earliest checkpoint wins ties
            best, best_epoch = val_ce, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        seconds = time.perf_counter() - tick
        elapsed += seconds
        history.append(dict(epoch=epoch, train_ce=loss_sum / seen, validation_ce=val_ce,
                            validation_acc=float((p.argmax(1) == y).mean() * 100), seconds=seconds))
        payload = dict(schema_version=1, identity=identity, epoch=epoch,
                       model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       best_model_state_dict=best_state, best_validation_ce=best,
                       best_epoch=best_epoch, rng=rng_state(gen), history=history, elapsed_seconds=elapsed)
        save_torch(last, payload)
        save_torch(out / "best_validation.pt", dict(schema_version=1, identity=identity,
                   epoch=best_epoch, validation_ce=best, model_state_dict=best_state))
        write_json(out / "training_progress.json", dict(completed_epochs=epoch, target_epochs=epochs,
                   best_epoch=best_epoch, best_validation_ce=best, elapsed_seconds=elapsed))
        logger.log("[epoch][seed={}] {}/{} trainCE={:.5f} valCE={:.5f} bestEpoch={} {:.2f}s".format(
            seed, epoch, epochs, loss_sum / seen, val_ce, best_epoch, seconds))
        if stop_after is not None and epoch == stop_after:
            return
    # Reconstruct summary on resume even if interruption followed epoch 100.
    saved = load_checkpoint(last, identity)
    if saved["epoch"] != epochs:
        raise RuntimeError("Training budget not completed")
    write_json(out / "history.json", saved["history"])
    summary = dict(completed_epochs=epochs, best_epoch=saved["best_epoch"],
                   best_validation_ce=saved["best_validation_ce"],
                   elapsed_seconds=saved["elapsed_seconds"],
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   peak_cuda_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2**20
                                           if device.type == "cuda" else None),
                   latest_sha256=sha256(last), best_sha256=sha256(out / "best_validation.pt"))
    write_json(out / "training_summary.json", summary)
    return summary


def evaluate(datasets, classes, device, model_dir, out, identity, logger):
    model_dir, out = Path(model_dir), Path(out)
    latest = load_checkpoint(model_dir / "latest.pt", identity)
    if latest["epoch"] != EPOCHS:
        raise RuntimeError("Formal evaluation requires exactly 100 completed epochs")
    best = load_checkpoint(model_dir / "best_validation.pt", identity)
    if best["epoch"] != latest["best_epoch"] or best["validation_ce"] != latest["best_validation_ce"]:
        raise RuntimeError("Incomplete checkpoint transaction; run train --resume first")
    if any(not torch.equal(v, latest["best_model_state_dict"][k]) for k, v in best["model_state_dict"].items()):
        raise RuntimeError("Best checkpoint differs from latest authoritative copy")
    model = IBGC(classes).to(device)
    model.load_state_dict(best["model_state_dict"])
    results, clean, clean_ids = {}, None, None
    for condition in ("clean", "mixed_1.0", "adaptive_1.0"):
        y, p, ids, ce = predict(model, datasets[condition], device)
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise RuntimeError("Invalid probability output")
        metrics = C.classification_metrics(y, p, classes)
        if condition == "clean":
            clean, clean_ids = metrics, ids
            small = C.compact_eval(metrics)
        else:
            if not np.array_equal(ids, clean_ids):
                raise RuntimeError("Attacked sample ID order differs from clean")
            small = C.attack_metrics(clean, metrics)
        small["cross_entropy"] = ce
        small["n"] = len(y)
        save_npz(out / (condition + "_probabilities.npz"), labels=y, probs=p,
                 sample_indices=ids, preds=p.argmax(1))
        small["probabilities_sha256"] = sha256(out / (condition + "_probabilities.npz"))
        results[condition] = small
        logger.log("[eval][{}] {}".format(condition, small))
    report = dict(identity=identity, completed_epochs=EPOCHS, best_epoch=best["epoch"],
                  checkpoint_sha256=sha256(model_dir / "best_validation.pt"), metrics=results)
    write_json(out / "evaluation.json", report)
    return report
