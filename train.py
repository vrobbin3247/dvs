import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import *
from model import *
import argparse
import os
torch.backends.cudnn.benchmark = True

# Parse command line arguments
parser = argparse.ArgumentParser(description='3D CNN Training for Drunk Detection')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--data_dir', type=str, default='/home/emb-vaibman/data/DIF/dataset_10sec', 
                    help='root directory of dataset')
parser.add_argument('--video_dir', type=str, default=None, 
                    help='directory containing videos (default: data_dir/all_videos)')
parser.add_argument('--csv_file', type=str, default=None,
                    help='path to CSV file (default: data_dir/video_files.csv)')
parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use (0.0-1.0, default: 1.0 uses all data)')
parser.add_argument('--use_class_weights', action='store_true',
                    help='use class weights to handle imbalanced dataset')
parser.add_argument('--save_dir', type=str, default='/home/emb-vaibman/data/DIF/dataset_10sec/saved_models/3d_cnn',
                    help='directory to save models')
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

# Save model path
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
print(f"Training Configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Data directory: {args.data_dir}")
print(f"  Save directory: {save_dir}")
print(f"  Data fraction: {args.data_fraction}")
print(f"  Use class weights: {args.use_class_weights}")
print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

#data loading - pass data paths from arguments
video_dir = args.video_dir if args.video_dir else os.path.join(args.data_dir, 'all_videos')
csv_file = args.csv_file if args.csv_file else os.path.join(args.data_dir, 'video_files.csv')

print(f"  Video directory: {video_dir}")
print(f"  CSV file: {csv_file}")

train_data = VideoDataset(mode="train", folder=video_dir + '/', file=csv_file) 
val_data = VideoDataset(mode="val", folder=video_dir + '/', file=csv_file)

# Use subset of data if requested
if args.data_fraction < 1.0:
    import random
    train_size = int(len(train_data) * args.data_fraction)
    val_size = int(len(val_data) * args.data_fraction)
    
    train_indices = random.sample(range(len(train_data)), train_size)
    val_indices = random.sample(range(len(val_data)), val_size)
    
    train_data = torch.utils.data.Subset(train_data, train_indices)
    val_data = torch.utils.data.Subset(val_data, val_indices)
    print(f"  Using {args.data_fraction*100}% of data")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
train_data_len = len(train_data)

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
val_data_len = len(val_data)

print(f"Training samples: {train_data_len}, Validation samples: {val_data_len}")

#model initialization and multi GPU
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model = model.to(device)
#loss, optimizer and scheduler
# Handle class imbalance with weighted loss
if args.use_class_weights:
    # Calculate class weights based on training data
    print("Calculating class weights for imbalanced dataset...")
    drunk_count = sum(1 for i in range(len(train_data)) if train_data[i][1] == 1.0)
    sober_count = train_data_len - drunk_count
    
    # Weight inversely proportional to class frequency
    total = drunk_count + sober_count
    weight_drunk = total / (2 * drunk_count) if drunk_count > 0 else 1.0
    weight_sober = total / (2 * sober_count) if sober_count > 0 else 1.0
    
    # For BCELoss, we'll apply weights in the training loop
    pos_weight = torch.tensor([weight_drunk / weight_sober]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Drunk: {drunk_count}, Sober: {sober_count}")
    print(f"  Class weight ratio (drunk/sober): {weight_drunk/weight_sober:.3f}")
    
    # Note: We need to modify model to output logits instead of probabilities
    print("  WARNING: Model needs to output logits (remove Sigmoid) when using BCEWithLogitsLoss")
else:
    criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=0.00001)
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

#resnet not pre-trained one
model = Model()

#using more than 1 GPU if available
#if (torch.cuda.device_count() > 1):
#   model = nn.DataParallel(model)
model = model.to(device)

#loss, optimizer and scheduler
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=0.00001)

best_acc = 0.0

print ("Training Started", train_data_len)
#training
for epoch in range(num_epochs):
	train_acc = 0.0
	epoch_loss = 0.0
	count = 0.0

	model.train()
	for i, (inputs,labels) in enumerate(train_loader):
		inputs = inputs.to(device) #change to device
		labels = labels.to(device)

		predictions = model(inputs) # predictions

		#now calculate the loss function
		loss = criterion(predictions.squeeze(), labels.float())
		
		#backprop here
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss += (loss.data * inputs.shape[0]) 
		predictions = predictions >= 0.5  # give abnormal label
		train_acc += (predictions.squeeze().float() == labels.float()).sum() #get the accuracy
	   
		#print('Ep_Tr:{}/{},step:{}/{},top1:{},loss:{}'.format(epoch, num_epochs, i, train_data_len //batch_size,train_acc1, loss.data))
	epoch_loss = (epoch_loss / float(train_data_len))
	train_acc = train_acc.float()
	train_acc /= float(train_data_len)
	
	print("Ep_Tr: {}/{}, acc: {}, ls: {}".format(epoch, num_epochs, train_acc.item(), epoch_loss.data))    


	#validation
	model.eval()
	epoch_loss = 0.0 
	val_acc = 0.0

	for i, (inputs,labels) in enumerate(val_loader):
		inputs = inputs.to(device) #change to device
		labels = labels.to(device)

		with torch.no_grad(): 
			predictions = model(inputs)

		loss = criterion(predictions.squeeze(), labels.float()) 
		epoch_loss += (loss.data * inputs.shape[0])   
		predictions = predictions >= 0.5  # give abnormal label
		
		val_acc += (predictions.squeeze().float() == labels.float()).sum()

	#print('Ep_vl: {}/{},step: {}/{},top1:{}'.format(epoch, num_epochs, i, test_data_len //batch_size,val_acc1))
	epoch_loss = (epoch_loss / float(val_data_len))
	val_acc = val_acc.float()
	val_acc /= float(val_data_len)
	
	print('Ep_vl: {}/{}, val acc: {}, ls: {}'.format(epoch, num_epochs, val_acc.data, epoch_loss.data))

	scheduler.step(epoch_loss.item()) #for the scheduler 
	

	if (best_acc <= val_acc.data):
		best_acc = val_acc.data
		state = {'acc':best_acc,'epoch': epoch+1, 'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}
		torch.save(state, os.path.join(save_dir, "best_3Dconv.pt"))
		print(f"✓ New best model saved! Accuracy: {best_acc:.4f}")
	
	print('Epoch: {}/{}, best_acc: {}'.format(epoch, num_epochs, best_acc)) #print the epoch loss
  
	if (epoch % 10) == 0:
	  state = {'epoch':epoch+1, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}
	  torch.save(state, os.path.join(save_dir, f"3Dconv_epoch{epoch}.pt"))
