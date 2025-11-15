import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.classification import Accuracy, MultilabelAccuracy


def diversity_loss_cosine(attention_maps, coeff=0.1):
    num_classes = attention_maps.shape[1]
    # attention_maps = attention_maps.view(batch_size, num_classes, -1)
    total_loss = 0.0

    counter = 0

    # Normalize attention maps along the last dimension to get unit vectors
    attention_maps_norm = F.normalize(attention_maps, p=2, dim=-1)

    # Compute cosine similarity for each pair of heads
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            counter += 1
            cos_sim = torch.einsum(
                "bij,bkj->bik",
                attention_maps_norm[:, i, :],
                attention_maps_norm[:, j, :],
            )

            # Since we want to penalize high similarity, we subtract from 1
            # This makes 1 (identical) bad and 0 (orthogonal) good
            diversity_penalty = 1 - cos_sim

            # Aggregate the penalty
            total_loss += diversity_penalty.sum()

    # Average the loss over batch size, sequence length, and number of comparisons
    # avg_loss = total_loss / (
    #     batch_size * npatchx * npatchy * num_classes * (num_classes - 1) / 2
    # )

    # print(avg_loss, total_loss)
    return coeff * total_loss / counter


def cosine_diversity_pytorch(attention_maps):
    num_classes = 151
    diversity_scores = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            vec1, vec2 = (
                attention_maps[:, i, :].view(-1),
                attention_maps[:, j, :].view(-1),
            )
            # Compute cosine similarity and adjust the range to [0, 1]
            diversity_score = (
                1 - F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)) / 2
            )
            diversity_scores.append(diversity_score)

    average_diversity = torch.mean(torch.stack(diversity_scores))
    return average_diversity.item()


# class DiversityLoss(object):
#     def __init__(self, num_heads, batch_size, coeff):
#         super(DiversityLoss, self).__init__()
#         self.num_heads = num_heads
#         self.batch_size = batch_size
#         self.coeff = coeff

#         self.I = Variable(
#             torch.zeros(self.batch_size, self.num_heads, self.num_heads)
#         ).cuda()
#         for p in range(self.batch_size):
#             for q in range(self.num_heads):
#                 self.I.data[p][q][q] = 1

#     def cal_loss(self, attention_map):
#         b, h, _, _ = attention_map[-1].shape
#         attention_map_r = attention_map[-1].view(b, h, -1)
#         attention_map_T = torch.transpose(attention_map_r, 1, 2).contiguous()

#         diversity_loss = self.Frobenius(
#             torch.bmm(attention_map_r, attention_map_T)
#             - self.I[: attention_map_r.size(0)]
#         )

#         return self.coeff * diversity_loss

#     def Frobenius(self, mat):
#         size = mat.size()
#         if len(size) == 3:  # batched matrix
#             ret = (torch.sum(torch.sum((mat**2), 1), 1) + 1e-10) ** 0.5
#             return torch.sum(ret) / size[0]
#         else:
#             raise Exception("matrix for computing Frobenius norm should be with 3 dims")


# class DiversityLoss(object):
#     def __init__(self, num_classes, batch_size, coeff):
#         super(DiversityLoss, self).__init__()
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.coeff = coeff

#         self.I = Variable(
#             torch.zeros(self.batch_size, self.num_classes, self.num_classes)
#         ).cuda()
#         for p in range(self.batch_size):
#             for q in range(self.num_classes):
#                 self.I.data[p][q][q] = 1

#     def cal_loss(self, attention_map):
#         attention_map = attention_map[:, :, : self.num_classes, self.num_classes :]
#         attention_map_r = attention_map[-1]
#         print(attention_map_r)
#         attention_map_T = torch.transpose(attention_map_r, 1, 2).contiguous()

#         diversity_loss = self.Frobenius(
#             torch.bmm(attention_map_r, attention_map_T)
#             - self.I[: attention_map_r.size(0)]
#         )

#         return self.coeff * diversity_loss

#     def Frobenius(self, mat):
#         size = mat.size()
#         if len(size) == 3:  # batched matrix
#             ret = (torch.sum(torch.sum((mat**2), 1), 1) + 1e-10) ** 0.5
#             return torch.sum(ret) / size[0]
#         else:
#             raise Exception("matrix for computing Frobenius norm should be with 3 dims")


class DiversityLoss(object):
    def __init__(self, num_classes, batch_size, coeff):
        super(DiversityLoss, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.coeff = coeff

        self.I = Variable(
            torch.zeros(self.batch_size, self.num_classes, self.num_classes)
        ).cuda()
        for p in range(self.batch_size):
            for q in range(self.num_classes):
                self.I.data[p][q][q] = 1

    def cal_loss(self, attention_map):
        # b, h, _, _ = attention_map.shape
        # attention_map_r = attention_map.view(b, h, -1) / torch.max(attention_map)
        # attention_map_T = torch.transpose(attention_map_r, 1, 2).contiguous()

        # bmm = torch.bmm(attention_map_r, attention_map_T)
        # bmm = bmm / torch.max(bmm)

        # diversity_loss = self.Frobenius(bmm - self.I[: attention_map_r.size(0)])

        b, h, _, _ = attention_map.shape
        attention_map_r = attention_map.view(b, h, -1)

        # Normalize each vector to unit length
        attention_map_r = F.normalize(attention_map_r, p=2, dim=2)

        attention_map_T = torch.transpose(attention_map_r, 1, 2).contiguous()

        # Perform batch matrix multiplication
        bmm = torch.bmm(attention_map_r, attention_map_T)

        # Calculate the diversity loss as the Frobenius norm of (bmm - I)
        diversity_loss = self.Frobenius(bmm - self.I[:b])

        return self.coeff * diversity_loss

    def Frobenius(self, mat):
        size = mat.size()
        if len(size) == 3:  # batched matrix
            ret = (torch.sum(torch.sum((mat**2), 1), 1) + 1e-10) ** 0.5
            return torch.sum(ret) / size[0]
        else:
            raise Exception("matrix for computing Frobenius norm should be with 3 dims")


class VisionTransformers(pl.LightningModule):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        num_classes,
        num_heads,
        batch_size,
        coeff=0.1,
        prune=False,
        diversify=False,
        lr_scheduler=False,
        task="single-label",
        multimodal=False,
        adaptive_dp_rate=False,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prune = prune
        self.diversify = diversify
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.multimodal = multimodal
        self.adaptive_dp_rate = adaptive_dp_rate

        if self.diversify:
            self.diversity_loss = DiversityLoss(
                num_classes=num_classes, batch_size=batch_size, coeff=coeff
            )

        if task == "single-label":
            self.train_metric = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_metric = Accuracy(task="multiclass", num_classes=num_classes)
        else:
            self.train_metric = MultilabelAccuracy(num_labels=num_classes)
            self.val_metric = MultilabelAccuracy(num_labels=num_classes)

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        if self.lr_scheduler:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            return self.optimizer

    def loss_function(self, logits, labels):
        return self.criterion(logits, labels)

    def class_specific_loss(self, logits, labels, num_classes):
        """
        Compute class-specific loss for each class token.

        Args:
            output_tokens (torch.Tensor): The output from the transformer model containing class tokens.
                                        Shape: [batch_size, num_classes, feature_dim]
            labels (torch.Tensor): Ground truth labels indicating the presence or absence of each class in the input.
                                Shape: [batch_size, num_classes]
            num_classes (int): Total number of classes.

        Returns:
            torch.Tensor: The class-specific loss for the batch.
        """

        # Create a meshgrid for the batch dimension
        batch_indices = torch.arange(logits.shape[0])  # Tensor: [0, 1, 2, ..., B-1]

        # Use advanced indexing to select the desired elements
        selected = logits[batch_indices, labels]

        loss = self.criterion(selected, labels)

        return loss

    def multi_token_cross_entropy_loss(self, predictions, targets, num_classes=16):
        # Convert targets to one-hot encoding to match predictions shape
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Expand targets to match predictions shape for direct comparison
        targets_expanded = targets_one_hot.unsqueeze(1).expand(
            -1, predictions.size(1), -1
        )

        # Calculate cross-entropy loss for each class token separately using .reshape()
        loss_per_token = F.cross_entropy(
            predictions.reshape(-1, num_classes),
            targets_expanded.reshape(-1, num_classes),
            reduction="none",
        )

        # Reshape back to [batch_size, num_class_tokens] to get per-token loss
        loss_per_token = loss_per_token.view(predictions.size(0), predictions.size(1))

        # Average (or sum) the loss across all class tokens
        loss = loss_per_token.mean(dim=1).mean()  # Average loss across batch and tokens

        return loss

    def get_penalty(self):
        penalty, sparsity_rate = 0, 0
        for layer_idx in range(self.model.mct.depth):
            penalty += self.model.mct.blocks[layer_idx].attn.gate.get_penalty()
            sparsity_rate += self.model.mct.blocks[
                layer_idx
            ].attn.gate.get_sparsity_rate()

            # penalty += self.model.transformer.layers[layer_idx][0].fn.gate.get_penalty()
            # sparsity_rate += self.model.transformer.layers[layer_idx][
            #     0
            # ].fn.gate.get_sparsity_rate()
        return penalty, sparsity_rate

    def training_step(self, train_batch, batch_idx):
        stats = {}

        y = train_batch["y"].float().cuda()
        y_ohe = train_batch["y_ohe"].cuda()

        data = train_batch["img"].float()

        if self.multimodal:
            s1 = train_batch["s1"].float()
            data = torch.cat([data, s1], 1)

        # output, img_attn, token_logits = self.forward(data)
        # output, img_attn = self.forward(data)
        # output = self.forward(data)
        # print(self.model.rate)
        # output = self.forward(data, y)
        if self.adaptive_dp_rate:
            if self.current_epoch < 20:
                self.model.rate = 0
            elif self.current_epoch >= 20 and self.current_epoch < 50:
                self.model.rate = 0.2
            elif self.current_epoch >= 50 and self.current_epoch < 40:
                self.model.rate = 0.3
            elif self.current_epoch >= 40 and self.current_epoch < 50:
                self.model.rate = 0.4
            else:
                self.model.rate = 0.5

        output, img_attn = self.model(data, y_ohe)

        patch_output = None
        # if len(output) == 2:
        #     output, patch_output = output

        train_token_loss = 0  # self.multi_token_cross_entropy_loss(token_logits, y)

        train_loss = self.loss_function(output, y) + train_token_loss

        # if self.current_epoch >= 20:
        if self.prune:
            penalty, sparsity_rate = self.get_penalty()
            train_loss = train_loss + penalty
            stats["sparsity_rate"] = sparsity_rate.item()

        if self.diversify:
            # stats["classification_loss"] = train_loss.item()
            # img_cosine = diversity_loss_cosine(img_attn)
            img_diversity = self.diversity_loss.cal_loss(img_attn)
            train_loss = train_loss + img_diversity
            # stats["diversity"] = img_diversity.item()
            # stats["cosine"] = img_cosine

        # if patch_output is not None:
        #     ploss = self.loss_function(
        #         patch_output, y
        #     )  # F.multilabel_soft_margin_loss(patch_output, y)
        #     train_loss = train_loss + ploss

        # train_metric = self.train_metric(output.mean(dim=1), y)
        train_metric = self.train_metric(output, y)

        stats["train_loss"] = train_loss.item()

        stats["train_accuracy"] = train_metric.item()

        self.log_dict(stats, on_step=False, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        # y = val_batch["y"].type(torch.LongTensor).cuda()
        y = val_batch["y"].float().cuda()
        y_ohe = val_batch["y_ohe"].cuda()

        data = val_batch["img"].float()

        if self.multimodal:
            s1 = val_batch["s1"].float()
            data = torch.cat([data, s1], 1)

        patch_output = None

        # output, _, _ = self.forward(data)
        # output, _ = self.forward(data)
        # output = self.forward(data)

        output, _ = self.model(data)

        # # val_loss = self.class_specific_loss(output, y, 16)

        # if len(output) == 2:
        #     output, patch_output = output

        val_loss = self.loss_function(output, y)
        # val_loss = F.multilabel_soft_margin_loss(output, y)

        if patch_output is not None:
            ploss = self.loss_function(
                patch_output, y
            )  # F.multilabel_soft_margin_loss(patch_output, y)
            val_loss = val_loss + ploss

        val_metric = self.val_metric(output, y)

        print(output[0], y[0])
        print(self.val_metric(output[0:1], y[0:1]))
        # val_metric = self.val_metric(output.mean(dim=1), y)

        stats = {"val_loss": val_loss, "val_accuracy": val_metric.item()}
        self.log_dict(stats, on_step=False, on_epoch=True, sync_dist=True)
