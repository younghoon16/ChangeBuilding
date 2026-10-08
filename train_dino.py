import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
from tqdm import tqdm
from data.WHU_MCI import (WHUBSDataset)
from data.LEVIR_MCI import (LEVIRCCDataset)
from data.S2Looking_MCI import (S2LookingDataset)
from data.SMARS_MCI import (SMARSDataset)
from data.LsSCD_MCI import (LsSCDDataset)
from model.model_encoder_dino import DINOv3FPNEncoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from model.blip2.CDCformer import CDCformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
import re

class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # 记录可用的GPU数量
        self.num_gpus = torch.cuda.device_count()
        print(f'Number of available GPUs: {self.num_gpus}')

        random_str = str(random.randint(10, 100))
        name = 'baseline_'+time_file_str() + random_str
        self.args.savepath = os.path.join(args.savepath, name)
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.4  # BLEU-4 score right now
        self.MIou = 0.4
        self.Sum_Metric = 0.4
        self.start_epoch = 0
        self.best_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)

        # Loss function
        # ===== Uncertainty Weighting =====
        self.log_sigma_det = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.log_sigma_cap = torch.nn.Parameter(torch.zeros(1, device=self.device))
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_det = torch.nn.CrossEntropyLoss().cuda()
        
        # Initialize / load checkpoint
        self.build_model()

        # Custom dataloaders
        if args.data_name == 'LEVIR_MCI':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        elif args.data_name == 'WHU_MCI':
            self.train_loader = data.DataLoader(
                WHUBSDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file,
                               args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                WHUBSDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file,
                               args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        elif args.data_name == 'S2Looking_MCI':
            self.train_loader = data.DataLoader(
                S2LookingDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                S2LookingDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        elif args.data_name == 'SMARS_MCI':
            self.train_loader = data.DataLoader(
                SMARSDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                SMARSDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
        elif args.data_name == 'LsSCD_MCI':
            self.train_loader = data.DataLoader(
                LsSCDDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LsSCDDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file,
                             args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.evaluator = Evaluator(num_class=3)

        self.best_model_path = None
        
    def _log_model_stats(self):
        print_log("=" * 60, self.log)
        print_log("Model Parameter Statistics", self.log)

        modules = {
            "Encoder (Backbone)": self.encoder,
            "Encoder Trans (Attention)": self.encoder_trans,
            "Decoder (Transformer)": self.decoder,
        }

        total_params = 0
        total_trainable = 0

        for name, module in modules.items():
            params = count_params(module, trainable_only=False)
            trainable = count_params(module, trainable_only=True)
            total_params += params
            total_trainable += trainable

            print_log(
                f"{name:30s} | Params: {params/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M "
                f"({trainable/params*100:.1f}%)",
                self.log
            )

        print_log("-" * 60, self.log)
        print_log(
            f"{'Total':30s} | Params: {total_params/1e6:.2f}M | Trainable: {total_trainable/1e6:.2f}M "
            f"({total_trainable/total_params*100:.1f}%)",
            self.log
        )

        # Uncertainty weighting params
        print_log(
            f"{'Uncertainty Weights':30s} | Params: 2 (log_sigma_det, log_sigma_cap)",
            self.log
        )

        print_log("=" * 60, self.log)
    
    def build_model(self):
        args = self.args
        self.encoder = DINOv3FPNEncoder(backbone_name=args.network, heads=args.num_heads)
        self.encoder_trans = AttentiveEncoder(n_layers=args.n_layers, backbone_name=args.network,
                                                feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                                heads=args.n_heads, dropout=args.dropout)
        self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                            vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                            word_vocab=self.word_vocab, n_head=args.n_heads,
                                            n_layers=args.decoder_n_layers, dropout=args.dropout)
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint)
            print('Load Model from {}'.format(args.checkpoint))
            pattern = r"epo_(\d+)_.*?MIou_(\d+)_Bleu4_(\d+)"
            match = re.search(pattern, args.checkpoint)
            if match:
                self.start_epoch = self.best_epoch = int(match.group(1))
                self.MIou = float(match.group(2)) / 1000
                self.best_bleu4 = float(match.group(3)) / 1000
                self.Sum_Metric = self.MIou + self.best_bleu4
                print(f"Checkpoint文件名解析成功: start_epoch={self.start_epoch}")
            else:
                print("Checkpoint文件名有误, 从零开始训练。")
            # self.best_bleu4 = checkpoint['bleu-4']
            self.decoder.load_state_dict(checkpoint['decoder_dict'])
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
            self.encoder.load_state_dict(checkpoint['encoder_dict'])
        
        if self.num_gpus > 1:
            print(f"🚀 Using {self.num_gpus} GPUs for DataParallel training")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.encoder_trans = torch.nn.DataParallel(self.encoder_trans)
            self.decoder = torch.nn.DataParallel(self.decoder)

        # Move to GPU, if available
        self.encoder = self.encoder.to(self.device)
        self.encoder_trans = self.encoder_trans.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr)
        uncertainty_params = [self.log_sigma_det, self.log_sigma_cap]
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=list(filter(lambda p: p.requires_grad, self.encoder_trans.parameters())) +
                uncertainty_params,
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr)
            
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0)
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0)
        self._log_model_stats()

    def training(self, args, epoch):
        self.encoder.train()
        self.encoder_trans.train()
        self.decoder.train()  

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        for id, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
            # if id == 120:
            #    break
            start_time = time.time()
            accum_steps = 64//args.train_batchsize

            # Move to GPU, if available
            imgA = imgA.to(self.device)
            imgB = imgB.to(self.device)
            seg_label = seg_label.to(self.device)
            token = token.squeeze(1).to(self.device)
            token_len = token_len.to(self.device)
            # Forward prop.
            feat1 = self.encoder(imgA)
            feat2 = self.encoder(imgB)
            feat1, feat2, seg_pre, _ = self.encoder_trans(feat1, feat2)
            
            # ------------------------------Caption Generate----------------------------------------------
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            
            # Calculate loss
            cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
            det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))
            precision_det = torch.exp(-self.log_sigma_det)
            precision_cap = torch.exp(-self.log_sigma_cap)

            loss = precision_det * det_loss \
                + self.log_sigma_det \
                + precision_cap * cap_loss \
                + self.log_sigma_cap
            # Back prop.
            loss = loss / accum_steps
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    # if epoch >10:
                    self.encoder_optimizer.step()

                # Adjust learning rate
                if self.decoder_lr_scheduler is not None:
                    self.decoder_lr_scheduler.step()
                # print(decoder_optimizer.param_groups[0]['lr'])
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    # if epoch > 10:
                    self.encoder_lr_scheduler.step()
                    # print(encoder_optimizer.param_groups[0]['lr'])

                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time #batch_time
            self.hist[self.index_i, 1] = det_loss.item() #train_loss
            self.hist[self.index_i, 2] = accuracy(seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
                                                    seg_label.reshape(-1), 1)
            self.hist[self.index_i, 3] = cap_loss.item()  # train_loss
            self.hist[self.index_i, 4] = accuracy(scores, targets, 5) #top5

            self.index_i += 1
            # Print status
            if self.index_i % args.print_freq == 0:
                print_log('Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Det_Loss: {4:.4f}\t'
                    'Det Acc: {5:.3f}\t'
                    'Cap_loss: {6:.5f}\t'
                    'Text_Top-5 Acc: {7:.3f}'
                    .format(epoch, id, len(self.train_loader),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,0])*args.print_freq,
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,1]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,2]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,3]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,4])
                                ), self.log)

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        self.evaluator.reset()
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                # Move to GPU, if available
                imgA = imgA.to(self.device)
                imgB = imgB.to(self.device)
                token_all = token_all.squeeze(0).to(self.device)
                # Forward prop.
                if self.encoder is not None:
                    feat1 = self.encoder(imgA)
                    feat2 = self.encoder(imgB)
                feat1, feat2, seg_pre, _ = self.encoder_trans(feat1, feat2)
                seq = self.decoder.sample(feat1, feat2, k=1)

                # for segmentation
                pred_seg = seg_pre.data.cpu().numpy()
                seg_label = seg_label.cpu().numpy()
                pred_seg = np.argmax(pred_seg, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(seg_label, pred_seg)
                # for captioning
                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % self.args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Fast test during the training
            # for segmentation
            Acc_seg = self.evaluator.Pixel_Accuracy()
            Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
            mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print_log(
                '\nDetection_Validation:\n' 'Acc_seg: {0:.5f}\t' 'Acc_class_seg: {1:.5f}\t' 'mIoU_seg: {2:.5f}\t' 'FWIoU_seg: {3:.5f}\t '
                .format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg), self.log)
            print_log('Iou: {}'.format(IoU), self.log)

            # Calculate evaluation scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        Sum_Metric = mIoU_seg + Bleu_4
        if Sum_Metric >= self.Sum_Metric:
            self.best_bleu4 = Bleu_4
            self.MIou = mIoU_seg
            self.Sum_Metric = max(Sum_Metric, self.Sum_Metric)
            #save_checkpoint
            print('Save Model')
            state = {'encoder_dict': self.encoder.state_dict(),
                    'encoder_trans_dict': self.encoder_trans.state_dict(),
                    'decoder_dict': self.decoder.state_dict()
                    }
            metric = f'Sum_{round(100000*self.Sum_Metric)}_MIou_{round(100000*self.MIou)}_Bleu4_{round(100000*self.best_bleu4)}'
            # metric = f'MIou_{round(10000 * self.MIou)}_Bleu4_{round(10000 * self.best_bleu4)}'
            model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
            best_model_path = os.path.join(self.args.savepath, model_name)
            torch.save(state, best_model_path)
            self.best_epoch = epoch
            self.best_model_path = best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data parameters
    parser.add_argument('--sys', default='linux', help='system win or linux')
    parser.add_argument('--data_folder', default='./data/WHU-BSCD-dataset/images/',help='folder with data files')
    parser.add_argument('--list_path', default='./data/WHU-BSCD/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/WHU-BSCD/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="WHU_MCI",help='base name shared by data files.')

    parser.add_argument('--gpu_ids', type=str, default='4,5,6,7', help='gpu ids in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=4, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1.25e-5, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multi-head gate')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    # Model parameters
    parser.add_argument('--decoder_type', type=str, default='Transformer', help='define the decoder type')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print(f"Using GPUs: {args.gpu_ids}")

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    trainer.args.checkpoint = None
    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
        trainer.training(trainer.args, epoch)
        trainer.validation(epoch)
        if epoch - trainer.best_epoch > 30 or epoch >= trainer.args.num_epochs:
            print(f'best_mIoU: {trainer.MIou:.4f}, best_BLEU-4: {trainer.best_bleu4:.4f}, best_epoch: {trainer.best_epoch}')
            break
            

