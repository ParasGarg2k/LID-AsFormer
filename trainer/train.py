import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model import MultiHeadAttLayer
from model import Encoder
from model import Decoder
from model import MyTransformer
from model import ConvFeedForward
from model import LIDTCNBlock
from utils import segment_bars_with_confidence
from utils import device
from utils import actions_dict
from utils import batch_gen
from loss import CombinedLoss
import torch.optim as optim

class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_head=4):
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_head)
        self.criterion = CombinedLoss(num_classes=num_classes, lambda2=0.15)
        self.num_classes = num_classes
        print(f"Model initialized with {num_head} attention heads.")
        print(f'Model Size: {sum(p.numel() for p in self.model.parameters()):,}')

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR: {}'.format(learning_rate))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct, total = 0.0, 0.0
            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                if batch_input.size(0) == 0: continue
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                
                ps = self.model(batch_input, mask)
                loss = self.criterion(ps, batch_target)
                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    _, predicted = torch.max(ps[-1].data, 1)
                    pred_len, target_len = predicted.size(1), batch_target.size(1)
                    if pred_len > target_len:
                        predicted, mask_acc, target_acc = predicted[:, :target_len], mask[:, 0, :target_len], batch_target
                    else:
                        predicted, mask_acc, target_acc = predicted, mask[:, 0, :pred_len], batch_target[:, :pred_len]
                    correct += ((predicted.cpu() == target_acc.cpu()).float() * mask_acc.cpu()).sum().item()
                    total += mask_acc.sum().item()

            scheduler.step(epoch_loss)
            batch_gen.reset()
            acc = float(correct) / total if total > 0 else 0
            avg_loss = epoch_loss / batch_gen.get_num_examples()
            print(f"[epoch {epoch + 1}]: loss = {avg_loss:.4f}, acc = {acc:.4f}")

            if (epoch + 1) % 5 == 0 and batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch)
                torch.save(self.model.state_dict(), f"{save_dir}/epoch-{epoch + 1}.model")
                torch.save(optimizer.state_dict(), f"{save_dir}/epoch-{epoch + 1}.opt")

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, False)
                if batch_input.size(0) == 0: continue
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)
                
                pred_len, target_len = predicted.size(1), batch_target.size(1)
                if pred_len > target_len:
                    predicted, mask_test, target_test = predicted[:, :target_len], mask[:, 0, :target_len], batch_target
                else:
                    predicted, mask_test, target_test = predicted, mask[:, 0, :pred_len], batch_target[:, :pred_len]
                
                correct += ((predicted.cpu() == target_test.cpu()).float() * mask_test.cpu()).sum().item()
                total += mask_test.sum().item()
        acc = float(correct) / total if total > 0 else 0
        print(f"---[epoch {epoch + 1}]---: tst acc = {acc:.4f}")
        self.model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            batch_gen_tst.reset()
            import time
            time_start = time.time()
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, False)
                if batch_input.size(0) == 0: continue
                vid = vids[0]
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = self.model(batch_input, mask)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    batch_target_squeezed = batch_target.squeeze()
                    if confidence.dim() == 0:
                        confidence, predicted = [confidence.item()], [predicted.item()]
                    else:
                        confidence, predicted = confidence.cpu().tolist(), predicted.cpu().tolist()

                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid.replace("/", "_"), i),
                                                 confidence, batch_target_squeezed.cpu().tolist(), predicted)

                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                if predicted.dim() == 0:
                    predicted = [predicted]
                
                for i in range(len(predicted)):
                    recognition.extend([list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate)
                
                f_name = vid.split('/')[-1].split('.')[0]
                with open(results_dir + "/" + f_name, "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write('\n'.join(recognition))
                    f_ptr.write('\n')
            time_end = time.time()
            print(f"Prediction completed in {time_end - time_start:.2f} seconds")