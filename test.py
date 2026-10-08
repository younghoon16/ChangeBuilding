import cv2
import torch.optim
from torch.utils import data
import argparse
import json
import torch.nn as nn
from tqdm import tqdm
from data.LEVIR_MCI import LEVIRCCDataset
from data.WHU_MCI import WHUBSDataset
from data.S2Looking_MCI import S2LookingDataset
from data.SMARS_MCI import SMARSDataset
from data.LsSCD_MCI import LsSCDDataset
from model.model_encoder_dino import DINOv3FPNEncoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
from utils_tool.visualizer import visualize_semantic_maps, visualize_dif_features
from thop import profile, clever_format
import time


class EncoderTransProfileWrapper(nn.Module):
    def __init__(self, encoder_trans, feat_shape=(512, 16, 16)):
        super().__init__()
        self.encoder_trans = encoder_trans
        self.feat_shape = feat_shape

    def forward(self, feat1, feat2):
        # 构造“假但对齐”的特征 list
        B = feat1.shape[0]
        C, H, W = self.feat_shape

        fake_feat = torch.randn(B, C, H, W, device=feat1.device)

        # 模拟 encoder_trans 期望的 list 输入
        img1_list = [fake_feat]
        img2_list = [fake_feat]

        return self.encoder_trans(img1_list, img2_list)
    
@torch.no_grad()
def model_profile(
    encoder,
    encoder_trans,
    decoder,
    img_shape=(1, 3, 256, 256),
    seq_len=20,
    device="cuda",
    warmup=10,
    repeat=100
):
    """
    统计 FLOPs / Params / FPS
    """
    encoder.eval()
    encoder_trans.eval()
    decoder.eval()

    dummy_A = torch.randn(img_shape, device=device)
    dummy_B = torch.randn(img_shape, device=device)
    
    encoder_trans_for_profile = EncoderTransProfileWrapper(
        encoder_trans,
        feat_shape=(512, 16, 16)  # 和 --feat_size 对应
    ).cuda()

    # ---------- FLOPs ----------
    # Encoder (shared)
    flops_enc, params_enc = profile(
        encoder, inputs=(dummy_A,), verbose=False
    )
    flops_enc *= 2  # A + B

    # Encoder Trans (cross attention + segmentation head)
    feat1 = encoder(dummy_A)
    feat2 = encoder(dummy_B)
    
    flops_trans, params_trans = profile(
        encoder_trans, inputs=(feat1, feat2), verbose=False
    )
    feat1, feat2, _, _ = encoder_trans(feat1, feat2)


    # Decoder (固定长度 caption)
    vocab_size = decoder.vocab_size
    dummy_token = torch.randint(
        0, vocab_size, (1, seq_len), device=device
    )
    flops_dec, params_dec = profile(
        decoder, inputs=(feat1, feat2, dummy_token, torch.tensor([seq_len], device=device)),
        verbose=False
    )

    total_flops = flops_enc + flops_trans + flops_dec
    total_params = params_enc + params_trans + params_dec

    flops_str, params_str = clever_format(
        [total_flops, total_params], "%.3f"
    )

    # ---------- FPS ----------
    for _ in range(warmup):
        feat1 = encoder(dummy_A)
        feat2 = encoder(dummy_B)
        feat1, feat2, seg_out, _ = encoder_trans(feat1, feat2)
        seq = decoder.sample(feat1, feat2, k=1)

    torch.cuda.synchronize()
    start = end3 = time.time()
    time_enc, time_trans, time_dec = 0, 0, 0
    
    for _ in range(repeat):
        feat1 = encoder(dummy_A)
        feat2 = encoder(dummy_B)
        end1 = time.time()
        time_enc += (end1 - end3)
        feat1, feat2, seg_out, _ = encoder_trans(feat1, feat2)
        end2 = time.time()
        time_trans += (end2 - end1)
        seq = decoder.sample(feat1, feat2, k=1)
        end3 = time.time()
        time_dec += (end3 - end2)
    torch.cuda.synchronize()
    end = time.time()

    
    fps = repeat / (end - start)
    fps_enc = repeat / time_enc
    fps_trans = repeat / time_trans
    fps_dec = repeat / time_dec

    return {
        "FLOPs-enc": flops_enc,
        "FLOPs-trans": flops_trans,
        "FLOPs-dec": flops_dec,
        "FLOPs": flops_str,
        "FPS": round(fps, 2),
        "Latency_ms": round(1000 / fps, 2),
        "FPS-enc": round(fps_enc, 2),
        "FPS-trans": round(fps_trans, 2),
        "FPS-dec": round(fps_dec, 2),
        "Latency_ms-enc": round(1000 / fps_enc, 2),
        "Latency_ms-trans": round(1000 / fps_trans, 2),
        "Latency_ms-dec": round(1000 / fps_dec, 2)
    }


def save_mask(pred, gt, name, save_path, args):
    # pred value: 0,1,2; map to black, yellow, red
    # gt value: 0,1,2; map to black, yellow, red
    name = name[0]
    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(gt, pred)
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    Miou_str = round(mIoU_seg, 4)
    # Miou_str save in json file named score
    json_name = os.path.join(save_path, 'score.json')
    if not os.path.exists(json_name):
        with open(json_name, 'a+') as f:
            key = name.split('.')[0]
            json.dump({f'{key}': {'MIoU': Miou_str}}, f)
        f.close()
    else:
        with open(os.path.join(save_path, 'score.json'), 'r') as file:
            data = json.load(file)
            key = name.split('.')[0]
            data[key] = {'MIoU': Miou_str}
        # write to json file
        with open(os.path.join(save_path, 'score.json'), 'w') as file:
            json.dump(data, file)
        file.close()

    # save mask
    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    # 消失：1 新增：2
    pred_rgb[pred == 1] = [0, 0, 255]
    pred_rgb[pred == 2] = [255, 0, 0]
    gt_rgb[gt == 1] = [0, 0, 255]
    gt_rgb[gt == 2] = [255, 0, 0]

    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + f'_mask.png'), pred_rgb)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_gt.png'), gt_rgb)
    # 保存image_A 和 image_B
    img_A_path = os.path.join(args.data_folder, 'test/A', name)
    img_B_path = os.path.join(args.data_folder, 'test/B', name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_A.png'), img_A)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_B.png'), img_B)


def save_captions(pred_caption, ref_caption, hypotheses, references, name, save_path):
    name = name[0]
    # return 0
    score_dict = get_eval_score([references], [hypotheses])
    Bleu_4 = score_dict['Bleu_4']
    Bleu_4_str = round(Bleu_4, 4)
    Bleu_3 = score_dict['Bleu_3']
    Bleu_3_str = round(Bleu_3, 4)

    # read JSON
    with open(os.path.join(save_path, 'score.json'), 'r') as file:
        data = json.load(file)
        key = name.split('.')[0]
        data[key]['Bleu_3'] = Bleu_3_str
        data[key]['Bleu_4'] = Bleu_4_str
    with open(os.path.join(save_path, 'score.json'), 'w') as file:
        json.dump(data, file)
    file.close()

    with open(os.path.join(save_path, name.split('.')[0] + f'_cap.txt'), 'w') as f:
        f.write('pred_caption: ' + pred_caption + '\n')
        f.write('ref_caption: ' + ref_caption + '\n')


def main(args):
    """
    Testing.
    """

    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint
    snapshot_full_path = args.checkpoint
    checkpoint = torch.load(snapshot_full_path)

    if args.visualize_path is not None and not os.path.exists(args.visualize_path):
        os.makedirs(args.visualize_path)

    args.result_path = os.path.join(args.result_path, os.path.basename(snapshot_full_path).replace('.pth', ''))
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    else:
        print('result_path is existed!')
        # clear folder
        for root, dirs, files in os.walk(args.result_path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    encoder = DINOv3FPNEncoder(backbone_name=args.network, heads=args.num_heads)
    encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                     feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                     heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                 vocab_size=len(word_vocab), max_lengths=args.max_length,
                                 word_vocab=word_vocab, n_head=args.n_heads,
                                 n_layers=args.decoder_n_layers, dropout=args.dropout)

    state_dict = checkpoint['encoder_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    encoder.load_state_dict(new_state_dict)
    state_dict = checkpoint['encoder_trans_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    encoder_trans.load_state_dict(new_state_dict)
    decoder.load_state_dict(checkpoint['decoder_dict'])
    # Move to GPU, if available
    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    # Custom dataloaders
    if args.data_name == 'WHU_MCI':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            WHUBSDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                         args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'S2Looking_MCI':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            S2LookingDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                         args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'SMARS_MCI':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            SMARSDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                         args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'LsSCD_MCI':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LsSCDDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                         args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.analysis:
        profile_info = model_profile(
            encoder=encoder,
            encoder_trans=encoder_trans,
            decoder=decoder,
            img_shape=(1, 3, 256, 256),  # 和你真实测试尺寸一致
            seq_len=20,
            device="cuda",
            warmup=10,
            repeat=200
        )

        print("=" * 60)
        print("Model Profiling Result")
        print("=" * 60)
        print(f"FLOPs-enc   : {profile_info['FLOPs-enc']}")
        print(f"FLOPs-trans : {profile_info['FLOPs-trans']}")
        print(f"FLOPs-dec   : {profile_info['FLOPs-dec']}")
        print(f"FLOPs       : {profile_info['FLOPs']}")
        print(f"FPS         : {profile_info['FPS']}")
        print(f"FPS-enc     : {profile_info['FPS-enc']}")
        print(f"FPS-trans   : {profile_info['FPS-trans']}")
        print(f"FPS-dec     : {profile_info['FPS-dec']}")
        print(f"Latency/img : {profile_info['Latency_ms']} ms")
        print(f"Latency-enc : {profile_info['Latency_ms-enc']} ms")
        print(f"Latency-trans : {profile_info['Latency_ms-trans']} ms")
        print(f"Latency-dec : {profile_info['Latency_ms-dec']} ms")
        print("=" * 60)
        return
        
    # Epochs
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0
    evaluator = Evaluator(num_class=3)
    with torch.no_grad():
        for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, name) in enumerate(
                tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()
            # decode_lengths = max(token_all_len.squeeze(0)).item()
            # Forward prop.
            if encoder is not None:
                feat1 = encoder(imgA)
                feat2 = encoder(imgB)
            feat1, feat2, seg_pre, dif_features = encoder_trans(feat1, feat2)
            #feat_expand = nn.Conv2d(128, 512, kernel_size=1).cuda()
            seq = decoder.sample(feat1, feat2, k=1)
            #seq = decoder.sample(feat1, feat2, k=1)
            # for segmentation
            pred_seg = seg_pre.data.cpu().numpy()
            seg_label = seg_label.cpu().numpy()
            pred_seg = np.argmax(pred_seg, axis=1)

            # for change detection: save mask?
            if args.save_mask:
                save_mask(pred_seg, seg_label, name, args.result_path, args)
            if args.visualize_path is not None:
                visualize_dif_features(dif_features, pred_seg, seg_label, name, args)
            # Add batch sample into evaluator
            evaluator.add_batch(seg_label, pred_seg)

            # for captioning
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'],
                                                                       word_vocab['<NULL>']}],
                                  img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "
            # for captioning: save captions?
            if args.save_caption:
                save_captions(pred_caption, ref_captions, hypotheses[-1], references[-1], name, args.result_path)
            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc + 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc + 1

        test_time = time.time() - test_start_time

        # Fast test during the training

        Acc_seg = evaluator.Pixel_Accuracy()
        Acc_class_seg = evaluator.Pixel_Accuracy_Class()
        mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
        F1 = evaluator.F1_Score_Class_Three()
        Recall = evaluator.Recall_Class_Three()
        Precision = evaluator.Precision_Class_Three()

        FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
        print(
            'Validation:\n' 'Acc_seg: {0:.5f}\t' 'Acc_class_seg: {1:.5f}\t' 'mIoU_seg: {2:.5f}\t' 'FWIoU_seg: {3:.5f}\t'
            .format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg))
        print(f'IoU:{IoU}, F1:{F1}, Recall:{Recall}, Precision:{Precision}')

        # Calculate evaluation scores
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.5f}\t' 'BLEU-2: {1:.5f}\t' 'BLEU-3: {2:.5f}\t'
                  'BLEU-4: {3:.5f}\t' 'Meteor: {4:.5f}\t' 'Rouge: {5:.5f}\t' 'Cider: {6:.5f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.5f}\t' 'BLEU-2: {1:.5f}\t' 'BLEU-3: {2:.5f}\t'
                  'BLEU-4: {3:.5f}\t' 'Meteor: {4:.5f}\t' 'Rouge: {5:.5f}\t' 'Cider: {6:.5f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Test of Captioning:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t'
              'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data parameters
    parser.add_argument('--sys', default='linux', help='system win or linux')
    parser.add_argument('--data_folder', default='D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images',
                        help='folder with image files')
    parser.add_argument('--list_path', default='./data/WHU-BSCD/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/WHU-BSCD/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="WHU_MCI", help='base name shared by data files.')
    parser.add_argument('--analysis', type=bool, default=False, help='if analysis the model')

    # Test
    parser.add_argument('--gpu_id', type=str, default='4', help='gpu ids in the training.')
    parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for test')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # save masks and captions?
    parser.add_argument('--save_mask', action='store_true', help='save the result of masks')
    parser.add_argument('--save_caption', action='store_true', help='save the result of captions')
    parser.add_argument('--result_path', default="./predict_result/",
                        help='path to save the result of masks and captions')
    parser.add_argument('--visualize_path', default=None, help='path to save middle result of masks and captions')
    # backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in multi-head gate')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")

    main(args)
