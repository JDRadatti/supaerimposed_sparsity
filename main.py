import copy
import os
import torch

from vgg import build_model, train, evaluate, create_dataloaders
from sparse import inject_switchable_masks, train_sparse

EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
DENSE_PATH = f"{BASE_DIR}/vgg_dense.pt"
SUPERIMPOSED_PATH = f"{BASE_DIR}/vgg_superimposed.pt"


def main():

    #
    # 0. Init
    #
    model = build_model()

    #
    # 1. Initial Training
    #
    if os.path.exists(DENSE_PATH):
        model.load_state_dict(torch.load(DENSE_PATH, weights_only=True))
    else:
        train(model, epochs=EPOCHS)
        torch.save(copy.deepcopy(model.state_dict()), DENSE_PATH)

    #
    # 2. Sparse Training
    #
    inject_switchable_masks(model, N=1, M=4, conv_axis="out", linear_orientation="row")
    model.set_sparsity_mode("sparse")
    train_sparse(model, epochs=EPOCHS)

    #
    # 3. Compare Accuracy
    #
    _, val_loader = create_dataloaders("./data", 128, 4)

    # Check 1:4 Accuracy
    model.set_sparsity_mode("sparse")
    _, sparse_acc = evaluate(model, val_loader, DEVICE)
    print(f"Final Sparse (1:4) Accuracy: {sparse_acc}%")

    # Check 1:1 (dense) Accuracy
    model.set_sparsity_mode("dense")
    _, dense_acc = evaluate(model, val_loader, DEVICE)
    print(f"Final Dense (1:1) Accuracy: {dense_acc}%")

    #
    # 4. Save final model
    #
    torch.save(model.state_dict(), SUPERIMPOSED_PATH)


if __name__ == "__main__":
    main()
