from slot_attention.params import SlotAttentionParams
from slot_attention.train import main

if __name__ == "__main__":
    params = SlotAttentionParams(max_epochs=150, num_train_images=10, num_val_images=10)
    main(params)
