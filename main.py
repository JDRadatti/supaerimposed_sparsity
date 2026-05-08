import copy
import os
import torch

from vgg import build_model, train, evaluate, create_dataloaders
from sparse import inject_switchable_masks

EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
DENSE_PATH = f"{BASE_DIR}/vgg_dense.pt"
SUPERIMPOSED_PATH = f"{BASE_DIR}/vgg_superimposed.pt"
TUNE_INVERSE_PATH = f"{BASE_DIR}/vgg_tuned_inverse.pt"
TUNE_FREEZE_BN_PATH = f"{BASE_DIR}/vgg_tuned_freeze_bn.pt"
_, VAL_LOADER = create_dataloaders("./data", 128, 4)


def check_accuracy(model, val_loader, device):
    model.set_sparsity_mode("sparse")
    _, sparse_acc = evaluate(model, val_loader, DEVICE)
    print(f"Sparse (1:4) Accuracy: {sparse_acc}%")

    model.set_sparsity_mode("dense")
    _, dense_acc = evaluate(model, val_loader, DEVICE)
    print(f"Dense (1:1) Accuracy: {dense_acc}%")


def main():

    #
    # 0. Init
    #
    model = build_model()

    #
    # 1. Initial Training
    #
    print("\n--- Phase 1: Dense (1:1) ---")
    if os.path.exists(DENSE_PATH):
        model.load_state_dict(torch.load(DENSE_PATH, weights_only=True))
    else:
        train(model, epochs=EPOCHS)
        torch.save(copy.deepcopy(model.state_dict()), DENSE_PATH)
    model = model.to(DEVICE)

    #
    # 2. Sparse Training
    #
    print("\n--- Phase 2: Sparse (1:4) ---")
    if os.path.exists(SUPERIMPOSED_PATH):
        inject_switchable_masks(
            model, N=1, M=4, conv_axis="out", linear_orientation="row"
        )
        model.load_state_dict(torch.load(SUPERIMPOSED_PATH, weights_only=True))
    else:
        inject_switchable_masks(
            model, N=1, M=4, conv_axis="out", linear_orientation="row"
        )
        model.set_sparsity_mode("sparse")
        train(model, epochs=EPOCHS)
        torch.save(model.state_dict(), SUPERIMPOSED_PATH)
    model = model.to(DEVICE)
    check_accuracy(model, VAL_LOADER, DEVICE)

    #
    # 3a. Tune inverse mask (BN unfrozen)
    #
    print("\n--- Phase 3a: Tune Inverse (BN unfrozen) ---")
    if os.path.exists(TUNE_INVERSE_PATH):
        inject_switchable_masks(
            model, N=1, M=4, conv_axis="out", linear_orientation="row"
        )
        model.load_state_dict(torch.load(TUNE_INVERSE_PATH, weights_only=True))
    else:
        model.set_sparsity_mode("tune_inverse")
        train(model, epochs=10)
        torch.save(model.state_dict(), TUNE_INVERSE_PATH)
    model = model.to(DEVICE)
    check_accuracy(model, VAL_LOADER, DEVICE)

    #
    # 3b. Tune inverse with BN frozen (starts from clean sparse checkpoint)
    #
    print("\n--- Phase 3b: Tune Inverse (BN frozen) ---")
    if os.path.exists(TUNE_FREEZE_BN_PATH):
        inject_switchable_masks(
            model, N=1, M=4, conv_axis="out", linear_orientation="row"
        )
        model.load_state_dict(torch.load(TUNE_FREEZE_BN_PATH, weights_only=True))
    else:
        inject_switchable_masks(
            model, N=1, M=4, conv_axis="out", linear_orientation="row"
        )
        model.load_state_dict(torch.load(SUPERIMPOSED_PATH, weights_only=True))
        model = model.to(DEVICE)
        model.set_sparsity_mode("tune_inverse_freeze_bn")
        train(model, epochs=10)
        torch.save(model.state_dict(), TUNE_FREEZE_BN_PATH)
    model = model.to(DEVICE)
    check_accuracy(model, VAL_LOADER, DEVICE)


if __name__ == "__main__":
    main()
