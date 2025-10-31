import os
import argparse
import tqdm 
import torch 
import random

from attack.utils import * 
from attack.ggs import GGS
from prettytable import PrettyTable
from torch.utils.data import DataLoader

# Set random seed
def set_seed(sd: int=17):
    seed = sd
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser(): 
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples') 
    # basic settings
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch') 
    parser.add_argument('--batchsize', default=8, type=int, help='the bacth size') 
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation') 
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation') 
    parser.add_argument('--zeta', default=2, type=float, help='') 
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack') 
    parser.add_argument('--num_neighbor', default=1, type=int, help='the number of neighbors') 
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model', 
                        choices=['resnet18', 'resnet50', 'resnet101', 'resnext50_32x4d', 'densenet121',
                                 'inception_v3', 'inception_v4', 'inception_resnet_v2',
                                 'vit_base_patch16_224','pit_b_224','visformer_small','swin_tiny_patch4_window7_224'])
    # mode settings
    parser.add_argument('--eval', action='store_true', help='evaluation mode (default: False)') 
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack (default: False)') 
    parser.add_argument('--targeted', action='store_true', help='targeted attack (default: False)') 

    # path settings
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data') 
    parser.add_argument('--output_dir', default='./results/', type=str, help='the path to store the adversarial patches') 
    parser.add_argument('--GPU_ID', default='0', type=str) 

    args = parser.parse_args()

    return args


def main(args):
    set_seed()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = ImageNet(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=True)

    if not args.eval:
        # Transfer attack mode
        if args.ensemble:
            args.model = ['resnet50', 'inception_v3', 'vit_base_patch16_224'] # example for ensemble attack
        attacker = GGS(model_name = args.model, 
                       epsilon = args.eps, 
                       targeted = args.targeted,
                       zeta = args.zeta,
                       epoch = args.epoch,
                       num_neighbor = args.num_neighbor)

        for images, labels, filenames in tqdm.tqdm(dataloader):
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
    else:
        # Evaluation mode
        avg = 0
        model_count = 0

        # Create table
        table = PrettyTable()
        table.field_names = ["Model Name", "ASR (%)"]
        
        # Helper function to evaluate model on dataloader
        def evaluate_model(model, dataloader, targeted, bias):
            """Evaluate model and return correct predictions count."""
            correct, total = 0, 0
            for images, labels, _ in dataloader:
                labels = labels[1] if targeted else labels
                pred = model(images.cuda())
                dim = 1 if len(pred.shape) == 2 else 0
                correct += (labels.numpy() == pred.argmax(dim=dim).detach().cpu().numpy() - bias).sum()
                total += labels.shape[0]
            return correct, total
        
        for model_name, model in load_pretrained_model(cnn_model_paper, inv_model_paper, vit_model_paper):
        # for model_name, model in load_pretrained_model(cnn_model_paper, inv_model_paper, vit_model_paper, ens_model_paper):
            bias = 1 if model_name in ens_model_paper else 0
            model = prepare_model(model)
            correct, total = evaluate_model(model, dataloader, args.targeted, bias)
            
            # Calculate ASR
            asr_value = (correct / total) * 100 if args.targeted else (1 - correct / total) * 100
            
            # Add row to table
            table.add_row([model_name, f"{asr_value:.2f}"])
            avg += asr_value
            model_count += 1

        # Print table
        avg_asr = avg / model_count
        table.add_row(["**Avg**", f"{avg_asr:.2f}"])
        print(table)

        # Save table results to file
        with open(os.path.join(args.output_dir, 'results_eval.txt'), 'a') as f:
            f.write(str(table) + '\n\n')

if __name__ == '__main__':
    args = get_parser()

    tail = '_targeted' if args.targeted else ('_ensemble' if args.ensemble else '')
    tail = '_targeted_ensemble' if (args.targeted and args.ensemble) else tail
    
    args.output_dir = os.path.join(args.output_dir, args.model + tail)
    main(args)
