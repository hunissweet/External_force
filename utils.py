
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
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
   # Setup a plot 
    plt.figure(figsize=(10, 5))
    for i in range(len(results_bunch)):
        results=results_bunch[i]
                   
        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Get the accuracy values of the results dictionary (training and test)


        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

     

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='train_loss_'+str(i))
        if i==len(results_bunch):
            plt.title('Train_Loss')
            plt.xlabel('Epochs')
            plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, test_loss, label='test_loss_'+str(i))
        if i==len(results_bunch):
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
    
    


    
def plot_prediction(Pred_Values,True_Values):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots Results

    Args: True value, Prediction results 
        
    """
    xlim =4
    ylim =4

    plt.figure(figsize=(12,6))
    plt.subplot(2,3,1)
    plt.plot(extraction(Pred_Values,0),label='Predict',color='blue',linestyle ="--")
    plt.plot(extraction(True_Values,0),label='True',color='blue')
    plt.ylim((-ylim, ylim))
    plt.legend(fontsize="12")
    
    plt.title('X')

    plt.subplot(2,3,2)
    plt.plot(extraction(Pred_Values,1),label='Predict',color='orange',linestyle ="--")
    plt.plot(extraction(True_Values,1),label='True',color='orange')
    plt.legend(fontsize="12")
    plt.ylim((-ylim, ylim))
    plt.title('Y')


    plt.subplot(2,3,3)
    plt.plot(extraction(Pred_Values,1),label='Predict',color='green',linestyle ="--")
    plt.plot(extraction(True_Values,1),label='True',color='green')
    plt.legend(fontsize="12")
    plt.ylim((-ylim, ylim))
    plt.title('Z')


    # For 



    plt.subplot(2,3,4)
    y = extraction(Pred_Values,0)
    x = extraction(True_Values,0)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('X')

    plt.subplot(2,3,5)
    y = extraction(Pred_Values,1)
    x = extraction(True_Values,1)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('Y')


    plt.subplot(2,3,6)
    y = extraction(Pred_Values,2)
    x = extraction(True_Values,2)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet,range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('Z')

    plt.tight_layout(pad=2)

    plt.show()

    
def predict(model: torch.nn.Module, 
            predict_data_loader: torch.utils.data.DataLoader):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in predict_data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output
def extraction(data,order):
    value=[]
    for i in range(len(data)):
        value.append(data[i][order])
    return value