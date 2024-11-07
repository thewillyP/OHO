#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import os 

choose = 2

# Load all .npz files
file_pattern = f'exp{choose}/trial*.npz'
script_dir = os.path.dirname(__file__)  # Get the directory of the current script
file_pattern = os.path.join(script_dir, file_pattern)  # Build the relative path
files = glob.glob(file_pattern)

def pad_sequences(sequences, maxlen=None, padding_value=np.nan):
    """Pads sequences to the same length."""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), maxlen), padding_value)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


# Initialize lists to hold data
tr_loss_list = []
vl_loss_list = []
te_loss_list = []
lr_list = []
dFdlr_list = []

# Load data from each file
for file in files:
    data = np.load(file)
    tr_loss_list.append(data['tr_loss'])
    vl_loss_list.append(data['vl_loss'])
    te_loss_list.append(data['te_loss'])
    lr_list.append(data['lr'])
    dFdlr_list.append(data['dFdlr_list'])

# Pad the loss arrays to ensure uniform shape
tr_loss_array = pad_sequences(tr_loss_list)
vl_loss_array = pad_sequences(vl_loss_list)
te_loss_array = pad_sequences(te_loss_list)
lr_array = pad_sequences(lr_list)
dFdlr_array = pad_sequences(dFdlr_list)

# Calculate mean and standard deviation for each loss
tr_loss_mean = np.nanmean(tr_loss_array, axis=0)
tr_loss_std = np.nanstd(tr_loss_array, axis=0)
vl_loss_mean = np.nanmean(vl_loss_array, axis=0)
vl_loss_std = np.nanstd(vl_loss_array, axis=0)
te_loss_mean = np.nanmean(te_loss_array, axis=0)
te_loss_std = np.nanstd(te_loss_array, axis=0)

# Overlay plot for Training and Test Mean Loss
plt.figure(figsize=(10, 6))
plt.plot(tr_loss_mean, color='blue', label='Mean Training Loss')
plt.fill_between(np.arange(len(tr_loss_mean)), 
                 tr_loss_mean - tr_loss_std, 
                 tr_loss_mean + tr_loss_std, 
                 color='blue', alpha=0.3)
plt.plot(te_loss_mean, color='green', label='Mean Test Loss')
plt.fill_between(np.arange(len(te_loss_mean)), 
                 te_loss_mean - te_loss_std, 
                 te_loss_mean + te_loss_std, 
                 color='green', alpha=0.3)
plt.title('Overlay of Mean Training and Test Loss with Variance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(np.percentile(np.concatenate((tr_loss_array, te_loss_array))[~np.isnan(np.concatenate((tr_loss_array, te_loss_array)))], 5),  
         np.percentile(np.concatenate((tr_loss_array, te_loss_array))[~np.isnan(np.concatenate((tr_loss_array, te_loss_array)))], 95))
plt.legend()

# Separate subplots for Training, Validation, and Test Loss with their mean
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Training Loss Plot
for tr_loss in tr_loss_array:
    axs[0].plot(tr_loss, alpha=0.3, color='blue')
axs[0].plot(tr_loss_mean, color='blue', linewidth=2, label='Mean Training Loss')
axs[0].fill_between(np.arange(len(tr_loss_mean)), 
                    tr_loss_mean - tr_loss_std, 
                    tr_loss_mean + tr_loss_std, 
                    color='blue', alpha=0.3)
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim(np.percentile(tr_loss_array[~np.isnan(tr_loss_array)], 5),  
                np.percentile(tr_loss_array[~np.isnan(tr_loss_array)], 95))
axs[0].legend()

# Validation Loss Plot
for vl_loss in vl_loss_array:
    axs[1].plot(vl_loss, alpha=0.3, color='orange')
axs[1].plot(vl_loss_mean, color='orange', linewidth=2, label='Mean Validation Loss')
axs[1].fill_between(np.arange(len(vl_loss_mean)), 
                    vl_loss_mean - vl_loss_std, 
                    vl_loss_mean + vl_loss_std, 
                    color='orange', alpha=0.3)
axs[1].set_title('Validation Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_ylim(np.percentile(vl_loss_array[~np.isnan(vl_loss_array)], 5),  
                np.percentile(vl_loss_array[~np.isnan(vl_loss_array)], 95))
axs[1].legend()

# Test Loss Plot
for te_loss in te_loss_array:
    axs[2].plot(te_loss, alpha=0.3, color='green')
axs[2].plot(te_loss_mean, color='green', linewidth=2, label='Mean Test Loss')
axs[2].fill_between(np.arange(len(te_loss_mean)), 
                    te_loss_mean - te_loss_std, 
                    te_loss_mean + te_loss_std, 
                    color='green', alpha=0.3)
axs[2].set_title('Test Loss')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Loss')
axs[2].set_ylim(np.percentile(te_loss_array[~np.isnan(te_loss_array)], 5),  
                np.percentile(te_loss_array[~np.isnan(te_loss_array)], 95))
axs[2].legend()

plt.tight_layout()


if choose == 2:

    # Calculate mean and standard deviation for training loss
    tr_loss_mean = np.nanmean(tr_loss_array, axis=0)
    tr_loss_std = np.nanstd(tr_loss_array, axis=0)

    # Training Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(tr_loss_mean, color='blue', label='Mean Training Loss')  # Plot mean training loss
    plt.fill_between(np.arange(len(tr_loss_mean)), 
                    tr_loss_mean - tr_loss_std, 
                    tr_loss_mean + tr_loss_std, 
                    color='blue', alpha=0.3, label='Variance')  # Fill variance area
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(np.percentile(tr_loss_array[~np.isnan(tr_loss_array)], 5),  
            np.percentile(tr_loss_array[~np.isnan(tr_loss_array)], 95))
    plt.legend()

    # Calculate mean and standard deviation for validation loss
    vl_loss_mean = np.nanmean(vl_loss_array, axis=0)
    vl_loss_std = np.nanstd(vl_loss_array, axis=0)

    # Validation Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(vl_loss_mean, color='orange', label='Mean Validation Loss')  # Plot mean validation loss
    plt.fill_between(np.arange(len(vl_loss_mean)), 
                    vl_loss_mean - vl_loss_std, 
                    vl_loss_mean + vl_loss_std, 
                    color='orange', alpha=0.3, label='Variance')  # Fill variance area
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(np.percentile(vl_loss_array[~np.isnan(vl_loss_array)], 5),  
            np.percentile(vl_loss_array[~np.isnan(vl_loss_array)], 95))
    plt.legend()


    # Calculate mean and standard deviation for test loss
    te_loss_mean = np.nanmean(te_loss_array, axis=0)
    te_loss_std = np.nanstd(te_loss_array, axis=0)

    # Test Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(te_loss_mean, color='green', label='Mean Test Loss')  # Plot mean test loss
    plt.fill_between(np.arange(len(te_loss_mean)), 
                    te_loss_mean - te_loss_std, 
                    te_loss_mean + te_loss_std, 
                    color='green', alpha=0.3, label='Variance')  # Fill variance area
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(np.percentile(te_loss_array[~np.isnan(te_loss_array)], 5),  
            np.percentile(te_loss_array[~np.isnan(te_loss_array)], 95))
    plt.legend()




# Learning Rate Trajectories
plt.figure(figsize=(10, 5))
for lr in lr_array:
    plt.plot(np.log(lr), alpha=0.3, color='purple')
plt.title('Learning Rate Trajectories')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')

# Gradient Trajectories with ylim adjustment
plt.figure(figsize=(10, 5))
for dFdlr in dFdlr_array:
    plt.plot(dFdlr, alpha=0.3, color='red')
plt.title('Gradient Trajectories')
plt.xlabel('Epochs')
plt.ylabel('Gradient')
plt.ylim(np.percentile(dFdlr_array[~np.isnan(dFdlr_array)], 5),
         np.percentile(dFdlr_array[~np.isnan(dFdlr_array)], 95))

# Show all plots
plt.show()
# %%
