def load_and_plot_losses(train_loss_file, val_loss_file):
    """
    Load and plot training and validation losses from a file.

    Args:
        loss_file (str): Path to the file containing loss values.

    """
    import matplotlib.pyplot as plt
    import numpy as np

    try: 
        # load from numpy file
        train_losses = np.load(train_loss_file)
        val_losses = np.load(val_loss_file)

        print(f"Training Losses: {train_losses}")

        # Plot the losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    date = "20250112-125505"
    train_loss_file = date + "/train_losses_vae_8_20250112-125505.npy"
    val_loss_file = date + "/val_losses_vae_8_20250112-125505.npy"
    load_and_plot_losses(train_loss_file=train_loss_file, val_loss_file=val_loss_file)