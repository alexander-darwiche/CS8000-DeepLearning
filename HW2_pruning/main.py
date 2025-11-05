from __future__ import print_function
import os
import sys
import logging
import argparse
import time
from time import strftime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import yaml

from vgg_cifar import vgg13

# settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='training batch size (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load-model-path', type=str, default="./model/cifar10_vgg13_acc_94.730.pt",
                    help='Path to pretrained model')
parser.add_argument('--sparsity-type', type=str, default='unstructured',
                    help="define sparsity_type: [unstructured, filter, etc.]")
parser.add_argument('--sparsity-method', type=str, default='omp',
                    help="define sparsity_method: [omp, imp, etc.]")
parser.add_argument('--yaml-path', type=str, default="./vgg13.yaml",
                    help='Path to yaml file')
                    

args = parser.parse_args()

# --- for dubeg use ---------
# args_list = [
#     "--epochs", "160",
#     "--seed", "123",
#     # ... add other arguments and their values ...
# ]
# args = parser.parse_args(args_list)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def get_dataloaders(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, download=True,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=256, shuffle=False)

    return train_loader, test_loader


# ============= the functions that you need to complete start from here =============

def read_prune_ratios_from_yaml(file_name, model):

        """
            This function will read user-defined layer-wise target pruning ratios from yaml file.
            The ratios are stored in "prune_ratio_dict" dictionary, 
            where the key is the layer name and value is the corresponding pruning ratio.

            Your task:
                Write a snippet of code to check if the layer names you provided in yaml file match the real layer name in the model.
                This can make sure your yaml file is correctly written.
        """

        if not isinstance(file_name, str):
            raise Exception("filename must be a str")
        with open(file_name, "r") as stream:
            try:
                raw_dict = yaml.safe_load(stream)
                prune_ratio_dict = raw_dict['prune_ratios']
                
                # ===== your code starts from here ======

                # Step 1: Get all layer names of the model (only consider CONV layers here)
                model_layer_names = [name for name, module in model.named_modules() if name != '' and isinstance(module, torch.nn.Conv2d)]
                print(model_layer_names)

                # Step 2: Check if the layer names in prune_ratio_dict match the model layer names
                for layer_name in prune_ratio_dict.keys():
                    base_name = layer_name.replace('.weight', '')
                    if base_name not in model_layer_names:
                        raise ValueError(f"Layer name '{layer_name}' in YAML file does not match any layer in the model.")

                # ===== your code ends here ======

                return prune_ratio_dict

            except yaml.YAMLError as exc:
                print(exc)


def unstructured_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    Implement magnitude-based unstructured pruning for weight tensor (of a layer)
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
  
    :return:
        torch.(cuda.)Tensor, pruning mask (1 for nonzeros, 0 for zeros)
    """
    ##################### YOUR CODE STARTS HERE #####################
    # Flatten and exclude already-zero weights
    nonzero_weights = tensor[torch.abs(tensor) > 0]

    # Step 1: Calculate how many weights should be pruned
    weights_to_prune = int(sparsity * tensor.numel())

    # Step 2: Find the threshold of weight magnitude (th) based on sparsity.
    weight_threshold = torch.topk(torch.abs(nonzero_weights).view(-1), weights_to_prune, largest=False).values.max()

    # Step 3: Get the pruning mask tensor based on the th. The mask tensor should have same shape as the weight tensor
    #         |weight| <= th -> mask=0,
    #         |weight| >  th -> mask=1
    mask = (torch.abs(tensor) > weight_threshold).float()

    # Step 4: Apply mask tensor to the weight tensor
    #         weight_pruned = weight * mask

    pruned_tensor = tensor * mask

    ##################### YOUR CODE ENDS HERE #######################

    # return the mask to record the pruning location ()
    return mask

def filter_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    implement L2-norm-based filter pruning for weight tensor (of a layer)
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
  
    :return:
        torch.(cuda.)Tensor, pruning mask (1 for nonzeros, 0 for zeros)
    """
    ##################### YOUR CODE STARTS HERE #####################

    # Flatten and exclude already-zero weights
    nonzero_weights = tensor[torch.abs(tensor) > 0]

    # Step 1: Calculate how many filters should be pruned
    num_filters = tensor.shape[0]
    filter_norms = torch.norm(tensor.view(num_filters, -1), p=2, dim=1)

    active_mask = (filter_norms > 0)
    active_norms = filter_norms[active_mask]

    num_filters_to_prune = int(sparsity * filter_norms.numel())
    
    # Step 2: Find the threshold of filter's L2-norm (th) based on sparsity.
    filter_threshold = torch.topk(active_norms, num_filters_to_prune, largest=False).values.max()      

    # Step 3: Get the pruning mask tensor based on the th. The mask tensor should have same shape as the weight tensor
    #         ||filter||2 <= th -> mask=0,
    #         ||filter||2 >  th -> mask=1
    mask = (filter_norms > filter_threshold).float().view(-1, 1, 1, 1)

    # Step 4: Apply mask tensor to the weight tensor
    #         weight_pruned = weight * mask
    pruned_tensor = tensor * mask

    ##################### YOUR CODE ENDS HERE #######################

    # return the mask to record the pruning location ()
    return mask

def apply_pruning(model, sparsity_type, prune_ratio_dict, masks):
    # calculate layer_wise prune ratio for current round (if IMP)

    # call unstructured_prune()  
    # or 
    # call filter_prune (...)
    # call unstructured_prune() for each layer
    for name, param in model.named_parameters():
        if name in prune_ratio_dict:
            sparsity = prune_ratio_dict[name]
            # call unstructured_prune() to get the mask
            # then apply the mask to the weight
            if sparsity_type == 'unstructured':
                mask = unstructured_prune(param, sparsity)
            elif sparsity_type == 'filter':
                mask = filter_prune(param, sparsity)

            param.data.mul_(mask)
            masks[name] = mask

    return model, masks

def test_sparsity(model, sparisty_type):
    
    # This function is used to check the model sparsity.
    # It should be able to print the sparisty ratio of each layer.

    # This example is obtained by testing a dense vgg13 model, 
    # this is why the sparity and number of zeros are all 0.
    # When you successfully pruned the model, then it should show the target sparisty ratio.
    # In other words, if the sparity of your pruned model is 0%, this indicates there must be something wrong.

    # features.x.weight is the layer name. 
    # You can the layer name and its weights by using the following for loop.
    # for name, weight in model.named_parameters():

    # For sparisty_type="unstructured":

    # Sparsity type is: xxxx (e.g., unstructured pruned or filter pruned)
    # (zero/total) weights of features.0.weight is: (0/1728). Sparsity is: 0.00%
    # (zero/total) weights of features.3.weight is: (0/36864). Sparsity is: 0.00%
    #       ...
    #       ...
    # ---------------------------------------------------------------------------
    # total number of zeros: 0, non-zeros: 9402048, overall sparsity is: 0.0000

    # For sparisty_type="filter":

    # (empty/total) filter of features.0.weight is: (0/64). filter sparsity is: 0.00%
    # (empty/total) filter of features.3.weight is: (0/64). filter sparsity is: 0.00%
    #       ...
    #       ...
    # ---------------------------------------------------------------------------
    # total number of filters: 2944, empty-filters: 0, overall filter sparsity is: 0.0000
    if sparisty_type == 'unstructured':
        print("Sparsity type is: unstructured pruned")
        total_zeros = 0
        total_weights = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight
                num_zeros = torch.sum(weight == 0).item()
                num_weights = weight.numel()
                layer_sparsity = 100. * float(num_zeros) / float(num_weights)
                total_zeros += num_zeros
                total_weights += num_weights
                print(f"(zero/total) weights of {name} is: ({num_zeros}/{num_weights}). Sparsity is: {layer_sparsity:.2f}%")
        overall_sparsity = 100. * float(total_zeros) / float(total_weights)
        print("---------------------------------------------------------------------------")
        print(f"total number of zeros: {total_zeros}, non-zeros: {total_weights - total_zeros}, overall sparsity is: {overall_sparsity:.4f}")
    elif sparisty_type == 'filter':
        print("Sparsity type is: filter pruned")
        total_empty_filters = 0
        total_filters = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight
                num_filters = weight.shape[0]
                filter_norms = torch.norm(weight.view(num_filters, -1), p=2, dim=1)
                num_empty_filters = torch.sum(filter_norms == 0).item()
                layer_sparsity = 100. * float(num_empty_filters) / float(num_filters)
                total_empty_filters += num_empty_filters
                total_filters += num_filters
                print(f"(empty/total) filter of {name}.weight is: ({num_empty_filters}/{num_filters}). filter sparsity is: {layer_sparsity:.2f}%")
        overall_sparsity = 100. * float(total_empty_filters) / float(total_filters)
        print("---------------------------------------------------------------------------")
        print(f"total number of filters: {total_filters}, empty-filters: {total_empty_filters}, overall filter sparsity is: {overall_sparsity:.4f}")
    return overall_sparsity

def masked_retrain(model, masks, optimizer, train_loader, test_loader, criterion):
    # when you fine-tune your pruned model, you only want to update the remaining weights (i.e., the weights that are not pruned),
    # while keeping the pruned weights to be 0.
    # A simple way to achieve this is:
    #   1. before update the weights, you find the pruning mask first.
    #   2. update all weights (including both remained and pruned weights).
    #   3. based on the pruning mask, prune the weights again.
    #      In this way, you can "keep" the pruned weights to be 0 after a training iteration.

    # Example:
    # For each training iteration
    #       ...
    #       optimizer.zero_grad()
    #       loss.backward()
    #       optimizer.step()
    #       # Here you may need a loop to loop over entire model layer by layer, then
    #       weight = weight * mask 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    acc = test(model, device, test_loader)
    print(f"Starting training accuracy: {acc}")
    for i in range(args.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Force the pruned weights to be zero
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        mask = masks[name]
                        param.data.mul_(mask)
        # Here you may need a loop to loop over entire model layer by layer, then
        if i % 10 == 0:
            model.eval()
            acc = test(model, device, test_loader)
            print(f"Epoch {i} training accuracy: {acc}")
    return model


def oneshot_magnitude_prune(model, sparsity_type, prune_ratio_dict, train_loader, test_loader, optimizer, criterion):
    # Implement the function that conducting oneshot magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # This function should also include the masked_retrain() function to conduct fine-tuning to restore the accurait 
    masks = {}
    model, masks = apply_pruning(model, sparsity_type, prune_ratio_dict, masks)
    test_sparsity(model, sparsity_type)
    model = masked_retrain(model, masks, optimizer, train_loader, test_loader, criterion)
    return model 

def iterative_magnitude_prune(model, sparsity_type, prune_ratio_dict, train_loader, test_loader, optimizer, criterion):
    # Implement the function that conducting iterative magnitude pruning
    # Target sparsity ratio dict should contains the sparsity ratio of each layer
    # the per-layer sparsity ratio should be read from a external .yaml file
    # You can choose the way to gradually increase the pruning ratio.
    # For example, if the overall target sparsity is 80%, 
    # you can achieve it by 20%->40%->60%->80% or 50%->60%->70%->80% or something else e.g., in LTH paper.
    # At each sparsity level, you need to retrain your model. 
    # Therefore, this IMP method requires more overall training epochs than OMP.
    # ** IMP method needs to use at least 3 iterations.
    iterations = 3

    # calculate layer-wise prune ratio for current round (if IMP)
    for name in prune_ratio_dict:
        prune_ratio_dict[name] = prune_ratio_dict[name] / iterations  # e.g., if total target sparsity is 60%, then each round prune 20%

    masks = {}
    for _ in range(iterations):  # e.g., 5 iterations
        model, masks = apply_pruning(model, sparsity_type, prune_ratio_dict, masks)
        test_sparsity(model, sparsity_type)
        model = masked_retrain(model, masks, optimizer, train_loader, test_loader, criterion)
    return model

def prune_channels_after_filter_prune(model, prune_ratio_dict, test_loader):
    # 
    # You need to implement this function to complete the following task:
    # 1. This function takes a filter pruned and fine-tuned model as input
    # 2. Find out the indices of all pruned filters in each CONV layer
    # 3. Directly prune the corresponding channels (that has the same indices) in next CONV layer (on top of the filter-pruned model).
    #    There is no need to fine-tune this model again.
    # 4. Return the newly pruned model

    # E.g., if you prune the filter_1, filter_4, filter_7 from the i_th CONV layer,
    # Then, this function will let you prune the Channel_1, Channel_4, Channel_7, from the next CONV layer, i.e., (i+1)_th CONV layer.

    # How to use this function:
    # 1. You will apply this function on a filter-pruned model (after fine-tune/mask retraine)
    # 2. There is no need to fine-tune the model again after apply this function
    # 3. Compare the test accuracy before and after apply this function
    #   
    # E.g., 
    #       pruned_model = your pruned and fine/tuned model
    #       test_accuracy(pruned_model)
    #       new_model = prune_channels_after_filter_prune(pruned_model)
    #       test_accuracy(new_model)

    # Answer the following questions in your report:
    # 1. After apply this function (further prune the corresponding channels), what is the change in sparsity?
    # 2. Will accuray decrease, increase, or not change?
    # 3. Based on question 2, explain why?
    # 4. Can we apply this function to ResNet and get the same conclusion? Why?
    
    model.load_state_dict(torch.load(args.load_model_path))
    test_accuracy_before = test(model, torch.device("cuda" if torch.cuda.is_available() else "cpu"), test_loader)
    test_sparsity_before = test_sparsity(model, 'filter')
    print(f"Test sparsity before pruning channels: {test_sparsity_before}")
    print(f"Test accuracy before pruning channels: {test_accuracy_before}")
    # Force the pruned weights to be zero
    with torch.no_grad():
        flag = False
        prev_mask = None

        for name, param in model.named_parameters():
            if name in prune_ratio_dict:  # Conv layer
                out_channels, in_channels, _, _ = param.shape
                
                if prev_mask is not None:
                    # Apply previous layer's mask along the input channels
                    param.data = param.data * prev_mask.view(1, -1, 1, 1)
                
                # Compute current layer's pruning mask
                filter_norms = torch.norm(param.view(out_channels, -1), p=2, dim=1)
                curr_mask = (filter_norms > 0).float().to(param.device)
                prev_mask = curr_mask
    
    test_accuracy_after = test(model, torch.device("cuda" if torch.cuda.is_available() else "cpu"), test_loader)
    test_sparsity_after = test_sparsity(model, 'filter')
    print(f"Test sparsity after pruning channels: {test_sparsity_after}")
    print(f"Test accuracy after pruning channels: {test_accuracy_after}")

    return model

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set up model archetecture and load pretrained dense model

    model = vgg13()
    model.load_state_dict(torch.load(args.load_model_path))
    if use_cuda:
        model.cuda()

    train_loader, test_loader = get_dataloaders(args)

    # Select loss function. You may change to whatever loss function you want.
    criterion = nn.CrossEntropyLoss()

    # Select optimizer. You may change to whatever optimizer you want.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # you may use this lr scheduler to fine-tune/mask-retrain your pruned model.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)

    # ========= your code starts here ========
    pre_prune_acc = test(model, device, test_loader)
    print(f"Pre-prune test accuracy: {pre_prune_acc}")
    prune_dict = read_prune_ratios_from_yaml(args.yaml_path, model)
    test_sparsity(model, args.sparsity_type)
    import pdb;pdb.set_trace()
    if args.sparsity_method in ['omp', 'imp']:
        if args.sparsity_method == 'omp':
            model = oneshot_magnitude_prune(model, args.sparsity_type, prune_dict, train_loader, test_loader, optimizer, criterion)
        elif args.sparsity_method == 'imp':
            model = iterative_magnitude_prune(model, args.sparsity_type, prune_dict, train_loader, test_loader, optimizer, criterion)
        else:
            raise Exception("sparsity_method not supported")
        
        # if args.sparsity_type == 'filter':
        #     model = prune_channels_after_filter_prune(model, prune_dict, test_loader)   

        sparsity = test_sparsity(model, args.sparsity_type)
        save_path = f"./model/cifar10_vgg13_{args.sparsity_type}_{args.sparsity_method}_{sparsity:.2f}_acc_{test(model, device, test_loader):.3f}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Pruned model saved to {save_path}")
    else:
        model = prune_channels_after_filter_prune(model, prune_dict, test_loader)
        sparsity = test_sparsity(model, args.sparsity_type)
        save_path = f"./model/cifar10_vgg13_{args.sparsity_type}_{args.sparsity_method}_{sparsity:.2f}_acc_{test(model, device, test_loader):.3f}_channelPruned.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Pruned model saved to {save_path}")
    """
        main()
            |- read_prune_ratios_from_yaml()
            |- IMP() or OMP()
                |-apply_pruning()
                    |-unstructured_prune()
                    |-filter_prune()
                |-masked_retrain()
    """

    # ---- you can test your model accuracy and sparity using the following fuction ---------
    # test_sparsity()
    # test(model, device, test_loader)

    # ========================================
    
if __name__ == '__main__':
    main()