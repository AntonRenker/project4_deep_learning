def load_and_plot_losses(loss_file):
    """
    Load and plot training and validation losses from a file.

    Args:
        loss_file (str): Path to the file containing loss values.

    """
    import matplotlib.pyplot as plt

    try:
        with open(loss_file, 'r') as f:
            lines = f.readlines()

        # Separate training and validation losses
        train_losses = []
        val_losses = []
        parsing_train = True

        for line in lines:
            line = line.strip()
            if line == "Validation Losses:":
                parsing_train = False
                continue
            if parsing_train and line != "Training Losses:":
                train_losses.append(float(line))
            elif not parsing_train:
                val_losses.append(float(line))

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
        print(f"Error: File {loss_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    load_and_plot_losses("loss_vae_8_20250111-023053.txt")