
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
  # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    

def plot_loss_curves(results_bunch):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...],}
    """
   # Setup a plot 
    plt.figure(figsize=(15, 7))
    for i in range(len(results_bunch)):
        results=results_bunch[i]
                   
        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

     

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, loss, label='train_loss_'+str(i))
        plt.hold(True)
        if i%2 !=0 or i==0:
            plt.title('Train_Loss')
            plt.xlabel('Epochs')
            plt.legend()
        
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, test_loss, label='test_loss_'+str(i))
        plt.hold(True)
        if i%2 !=0 or i==0:
            plt.title('Test_Loss')
            plt.xlabel('Epochs')
            plt.legend()
        
    plt.show()

class Data:
    def __init__(self, X, y,sequence_length=1):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]
    
    def __len__(self):
        return len(self.X)


    
