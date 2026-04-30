import phase1_dense as p1
import phase2_masks as p2
import phase3_sparse as p3
import phase4_mixed as p4

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
CHECKPOINT_INTERVAL = 5
N, M = 1, 4  # 1:4 sparsity


def main():
    p1.run(SEED, EPOCHS, BATCH_SIZE, CHECKPOINT_INTERVAL)
    p2.run(N, M)
    p3.run(epochs=EPOCHS, lr=0.01, checkpoint_interval=CHECKPOINT_INTERVAL)
    p4.run(epochs=EPOCHS, lr=0.01, checkpoint_interval=CHECKPOINT_INTERVAL)


if __name__ == "__main__":
    main()
