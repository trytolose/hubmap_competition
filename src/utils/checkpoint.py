from collections import deque
from pathlib import Path
import torch


class CheckpointHandler:
    def __init__(self, model, chks_path, max_count=5):

        self.chks_path = Path(chks_path)
        self.best_score = 0
        self.max_count = max_count
        self.model = model
        self.chks_path.mkdir(parents=True, exist_ok=True)

    def update(self, epoch, cur_score):
        if cur_score > self.best_score:
            self.best_score = cur_score
            print("+", end="")
            chk_filename = self.chks_path / f"epoch_{epoch}_score_{cur_score:.4f}.pth"
            current_chks = sorted(list(self.chks_path.glob("*.pth")), key=lambda x: x.stem.split("_")[-1])


            if len(current_chks) >= self.max_count:
                try:
                    current_chks[0].unlink()
                except OSError as e:
                    print("Error: %s : %s" % (current_chks[-1], e.strerror))
            torch.save(self.model.cpu().state_dict(), chk_filename)

