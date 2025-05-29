import random, numpy as np, torch, datetime, logging
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_logger():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"train_{ts}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    def _log(msg: str):
        print(msg)
        logging.info(msg)
    _log(f"Log file: {log_file}")
    return _log
