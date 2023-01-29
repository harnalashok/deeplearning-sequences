# Last amended: 26th July, 2019
# My folder: /home/ashok/Documents/14.sensor_data
# VM: lubuntu_deeplearning_I.vdi
# Ref: Page 207, Chapter 6, Deep Learning with Python by Fracois Chollete
# Download dataset from:
# 1. Link to my google drive
#  https://drive.google.com/file/d/1rnhlFKmmmhXqawaIBgjSTsqGrTLCUldV/view?usp=sharing
# 2. Link to original datasource
#  https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip

# Objectives:
#             i)  Working with timeseries data
#             i)  Working with sensor data
#                 (Data comes from many sensors)
#             ii) Processing data to make it fit for modeling
#            iii) Creating a data generator for training and validation
#            iv)  Making predictions using
#                   a) Fully connected dense model
#                   b) GRU model
#                   c) GRU model with dropouts
#                   d) Stacked GRU models
#                   e) Bidirectional RNN layer
#
#

# We will predict temperature
# Sensor data is recorded every 10 minutes. So per-day we have:
#   no of minutes:              24 * 60     =  1440
#   no of 10 minutes interval: (24 * 60)/10 = 144 datapoints/per day
#   no of data-points in 10 days: 1440

# 	source activate tensorflow
#   source deactivate tensorflow


# Reset all variables
%reset -f

# 1.0 Call libraries
import numpy as np
import matplotlib.pyplot as plt
import os, time, gc


####### How to read a csv file directly in numpy ###########

# 1.1 Where is my data?
data_dir = '/home/ashok/.keras/datasets/jena_climate'

# 1.2 Join datapath with filename (intelligently)
#     If you are on Windows, assign to fname full
#     data path+ filename
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
fname

# 1.3 Read datafile, line by line
# 1.3.1 First get a handle to file
f = open(fname)   # open() default mode is text+read

# 1.3.2 Use handle to read complete file
data = f.read()


# 1.3.3 Close file
f.close()

# 1.3.4 Observe data
type(data)        # str
data[0:200]      # Whole data is read as one string
                 # Read first 200 characters of string
                 # Note '\n' at 196th character


# 1.4 Look at data
# 1.4.1 Split data on newline (\n). So how many records?
lines = data.split('\n')    # Split each line at 'newline'
type(lines)                 # list =>  All lines are in one list
len(lines)                  # 420552
type(lines[50])             # Each element of list is still a 'str'


# 1.4.2 Does any header exist? Check
lines[0]                   # yes, it does
lines[1]


# 1.4.3 Extract header (field-names)
header = lines[0].split(',')  # Split at each ','
header

# 1.4.4 How many columns/fields?
cols = len(header)
cols             # 15


# 1.4.5 Print first three rows.
#       Note 10-minute gap in the
#       observations
lines[1:4]     # A list of 3 string elements

len(lines)    # 420552 or header + 420551 data points

totalsamples = len(lines) - 1    # 420551 (exclude header)

# 2.0 Convert all 420551 data points into a numpy array
#     for processing

# 2.1 First create a zero-valued 2D-array
#      While creating zero-valued 2D-array, forget
#        first column or time data
#     So array size will be same as that of data
#     We will also forget 1st column
float_data = np.zeros((totalsamples, cols -1 ))  # Exclude 1st date/time col
float_data.shape           # (420551,14)


# 2.2 Fill this 2D-zero-valued array, row-by-row using for-loop
# 2.2.1 First get an array of 420551 values
#       0 ,1, 2, 3,...420550
numbList=np.arange(len(lines) - 1 )
numbList

# 2.2.2 See how a line is split in respective
#       field values. We want split values to
#       be an array. But after the split,
#       they are a list
x = lines[1].split(',')
type(x)      # list
# 2.2.3
x = np.asarray(x[1:], dtype = 'float32') # Exclude 1st date/time column
type(x)
x

# 2.3  Fill up zero-array,row-by-row, with sensor data
for i in numbList:      # ie uptil the last line
    # 2.3 Now do this for all lines using for-loop
    row = lines[i+1].split(',')     # i starts from 0 but we ignore header
    # 2.3.1 'row' is a list. Select all but 1st element
    row= row[1:]                    # Ignore the date column
    values = np.asarray(row, dtype = 'float32')
    float_data[i, :] = values      # Fill zero-array, row-by-row

# 2.3.2 Check
float_data.shape     # (420551,14)

# 2.3.2
float_data[0]


## Plotting temperature
##*********************

# 3. Let us plot temperature, the IInd column
#    Check 'header', if you like
# 3.1 Get column with index 1
temp = float_data[:, 1]
temp

# 3.2 plot it. It is highly periodic
plt.plot(range(len(temp)), temp)
plt.show()

# 3.3 Let us see 12-hour variation
#     How many readings in one-day?  (24 * 60 )/10 = 144
#     So half day = 72 points
#     It is half of this variation we want to predict

plt.plot(range((12*60//10)), temp[:72])
plt.show()

# 3.4 Delete unwanted variable
del temp
gc.collect()

### Data pre-processing
###---------------
# 3.4 For processing we center and scale all data
#     Could have used sklearn's StandardScaler()
#     We decide upon a max sample size of 2 lakh
#     Were we rich in memory, more could have been selected

training_data_size = 420551     # select 200000 to save memory

# 3.4.1 Get mean of every column
#       We extract mean from training data only but apply
#       to whole of data
mean = float_data[:training_data_size].mean(axis = 0)
mean[:4]     # Show Ist four column means
# 3.4.2 Subtract from each column, its respective mean
#       We subtract this from all data.
float_data = float_data - mean
# 3.4.3 Get std deviation of each column but from training data only
std = float_data[:training_data_size].std(axis = 0)
std[:4]    # First four column std deviation
# 3.4.4 Divide each column by its respective std deviation
float_data /= std
float_data.shape

################## Learning to create data generators ###############
##################### Expt & explanation #############################
## 4.0 Problem defined:
## --------------------
## Starting from any point, say, X, in dataset, our intention is
##  to lookback 1440 timesteps (10days) behind, and predict temp
##   one day ahead of X ie 144 time steps ahead of X.
##    Considering redundancy in data, instead of considering all
##     1440 lookback points, we will pick just 240 datapoints at
##      hourly interval, (ie 1440/6 = 240).
##      (144 timesteps = 144 * 10 = 1440 minutes = 24 * 60)
##
## Train Data Generator
## ---------------------
## 4.1 Case 1: How will we generate our training data?
##  We will treat this as an image classification problem (sort of).
##  For every image, we have a two-dimensional data. Here we want
##  to predict temperature, but in pur data, besides temp, we
##  also have other parameters such as dew, humidity, pressure
##  etc
##  So, pick a random X and then we will generate our data, as:
##        Batch 1 (240 observations)
##                (14 attributes + target)
##    obs   pressure     temp    humidity        Target
##     1     0.87        0.1     0.23
##     2     0.28        0.91    0.77             0.63
##     3     0.56        0.33    0.99
##    ..           ..      ..
##    ..           ..      ..
##  In a normal timeseries, we would write data in 1D form:
##    0.1, 0.91, 0.33     ..... and predict 0.63.

##  Per epoch, we will pick 128 random Xs and thus generate
##  128 batches of data (steps_per_epoch = 128). Our data generator
##  can generate infinite batches of images, epoch-after-epoch.
##
##  Also just as we have batches of images for making predictions,
##  we will have heve 128 batches of (240 X 14) datasets. For
##  training data, all our 128 batches will be picked up randomly
##  from first 2lakh data points. We select some arbitrary points
##  (arbitrary Xs) within our data and from there pick up 240 rows.
##  We will make this random selection of 128 points many
##  times (infinitely) in our training-data-generator (steps_per_epoch).
##  This will help in changing the sequence of time steps:
##
##  Example of an epoch: steps_per_epoch=2
##  Our training generator is configured to generate each time a batch of ONE.
##
##  next(train_gen):
##  First batch of 3 senetences, 7 words and vocab of 32 words:
##               I am an Indian. I know Hindi    7 words, each 32 oneHotEncoded
##               I like to eat simple spicy food.
##               I will visit France this summer end.
##
##  next(train_gen):
## IInd batch of shape (3 X 7 X 32 ):
##               I am Punjabi. I stay in Punjab        # Randomly selected
##               I will visit France this summer end.  # Randomly selected
##               I like to eat simple spicy food.      # Randomly selected

##
##  Here is exactly how a batch is created for training:
##  ------------------------------------------------------
##  Starting from any point between row 1440 till last row that is 200000th row,
##  we make some random selection of numbers. Say, one number is
##  20451. From this number we look back at all the data 1440 timesteps
##  behind that is we start from 20451 - 1440 = 19011 . And start picking up,
##  every point 6th point this will make us available 1440/6 = 240 observations.
##  And we want to predict temperature 'delay' ahead. So our target
##  value for this 2D data will be temperature reading, at row: 20451 + delay
##
## 4.2 Validation data Generator
## -----------------------------
##  For validation data, we start with min_index = 200001 and keep max_index = 300000.
##  This time we will not pick up batches randomly. We start with 200001 + 1440 = 2001441
##   and get 128 numbers (including 2001441) serailly that is:
##     201441, 201442,...201568 (200001 + 1440 +128 - 1). We take each of these
##      numbers and prepare dataset of 240 points (each point at an interval of 6steps)
##       Thus we have a first batch of 128.
##
## 4.3 Test Data generator
##  ----------------------------------------------
#  Same as for validation but this time min_index = 300001 and max_index = None
#   that is upto: len(float_data) - delay - 1


# 4.4. Define some common constants
lookback = 1440  # timesteps. (same as 10 days) In RNN, we will lookback at
                 #            1440 timesteps. Each timestep is 10 minutes.
# 4.5
step = 6         # timesteps.  But we will pick temp values at every
                 # hour rather than ever 10 minutes that is pick one point after
                 #   every 6th point.
                 # So in this per lookback period datapoints will be 1440/6 = 240
                 # Or for every 'image' or 'dataset' we have image size
                 #  (ie dataset size) of 240 X 14.
# 4.6
delay = 144      # timesteps. (same as 24 hours) We will forecast for a temperatue
                 # 144 timesteps ahead
# 4.7
batch_size = 128 # Analyse 'dataset' in batches of 128 considering memory limitations
                 # So we will analyse in memory at a time: 128 (samples) X 240 X 14


#############################
###### Case 1   Training data generator
#############################

# Define constants specific to this Case that is training data
# 5.
min_index = 0        # For training, we start from here
max_index = 200000   # Our last point for training
shuffle = True       # Training batches will be picked up randomly


##******************************************************
# 6. Begin creating batches of datasets--One batch only
##******************************************************

# 6.1 Our 'X' should be between min_index and max_index.
if max_index is None:    # None means no value is specified. We then calculate
                         # None is same as NULL in other languages
    max_index = len(float_data) - delay - 1   # 420406. Keep it just less than forecasting duration

# 6.2
i = min_index + lookback     # Begin here so that we can pick up earlier points

# 6.3
i      # 1440 timesteps to start with or 10 days lookback
       # If this is time, t, RNN gets data for t-1,
       #  t-2, right upto first point as lookback is 1440 timesteps
       # np.random.randint(low, high, howmany)
       # np.random.randint(0,100,10)   Gives 10 random points

# 6.4
#--
if shuffle:               # shuffle = True for training only
    # Generate 128 random points (Xs) for picking training batches
    #   Try: np.random.randint(0, 100, size = 12)

    #                             1440              200000           128
    rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
else:                       # for validation and test datasets
    if i + batch_size >= max_index:   # Given some 'i', if it exceeds limit
        i = min_index + lookback      #  then set it to min possible
    # Try np.arange(1, min(1+10,14))
    rows = np.arange(i, min(i + batch_size, max_index))  # Get a list of all
                                                         #  points from i uptill the limit
    i += len(rows)      # Increment i by 128
#--

# 6.5 Have a look at rows from where to start picking up 240 lookback points
rows


# 6.6 Given row-points, get samples of datasets, each of 240 X 14
#     Now get an array of zeros of requistie batch size:
# Try np.zeros((2,3,4))

#  3D array          128            240         14
samples = np.zeros((len(rows),lookback // step, float_data.shape[-1]))
samples = np.zeros((128,240,14))    # same as above
# 6.7 Check
samples.shape   # (128,240,14) (batch size, no_of_points_per_hour, attributes)


# 6.8 There will be as many targets as there are batches
targets = np.zeros((len(rows),))
targets = np.zeros((128,))     # same as above
targets
len(targets)    # 128

# 7. Prepare to fill in first batch into our 3D zero-array
j = 0                 # Later we can loop on this 'j'
row = rows[j]         # First random point
row

# 7.1 Whereever I am standing (say at, 1000), I go behind
#     'lookback' timesteps (say 100). From there, I proceed
#      forward, every 6-datapoints upto where I am standing
#       (ie upto 1000). So I collect 240 datapoints:
#        Try: indicies = list(range(1000-100, 1000, 6))

#  7.2 Get indices of those 240 datapoints at intervals of 6 (step)
#      beginning from  jth index of rows[j]
indices = range(rows[j] - lookback, rows[j], step)   # 1440 timesteps values but in steps of 6
indices   # range(195492, 196932, 6)
len(indices)      # total: 240 points
list(indices)     # What are the points. Spaced at interval of 6

# 8. Finally, fill our zero-sample with data at these indicies
samples[j] = float_data[indices]          # 240 points
samples[j].shape                          # 240 X 14

# 8.1 Just have a look
samples[j]

# 8.2 Our prediction datapoint is 'delay' timesteps ahead
#     And in the data, temperature is at index 1
targets[j] = float_data[rows[j] + delay][1]   # Taregt value is the value in IInd column

# 8.3                                         #  at the end of rows[j] + delay
targets[j]


#############################
###### Case 2   Validation data generator
#############################

# 9. We will have min_index = 200001 and max_index = 300000 and shuffle = False
#    Other constants remain same:
min_index = 200001
max_index = 300000
shuffle = False

# 9.1
if max_index is None:
    max_index = len(float_data) - delay - 1      # 420406

# 9.2
i = min_index + lookback

i      # 201441

# 9.3   We have to get 128 consecutive points from where to pickup
#       128 batches of 240 X 14
if shuffle:
    rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
else:
    # 9.3.1
    if i + batch_size >= max_index:  # If unable to get list of 128 points ahead
                                     #   to get 128 batches of (240 X 14)
        i = min_index + lookback     #  start afresh. min_index for validation is 200001
    # 9.3.2
    rows = np.arange(i, min(i + batch_size, max_index))  # Get a list of all
                                                         #  points from i uptill the limit
   # 9.3.3
    i += len(rows)                     # Advance i by 128 steps to get next 128 batches

# 10
i              # Next batch begins at: 200001 + 1440 + 128 = 201569
len(rows)                    # 128
rows           # Just list them

# 11. Get an array of batch-size of all zeros
# Now get
#                      128            240         14
samples = np.zeros((len(rows),lookback // step, float_data.shape[-1]))
samples.shape   # (128,240,14) (batch size, no_of_points_per_hour, attributes)

# 11.1 There will be as many targets as there are batches
targets = np.zeros((len(rows),))
targets
len(targets)    # 128

# 12. Fill the batch with data
row = rows[0]    # First data point to start first batch from
j = 0
row              # 201441

# 12.1 Whereever I am standing, I go loopback back
#      From there, I proceed forward, every 6-datapoints
#      So I collect 240 datapoints
#  Get indices of those 240 datapoints

#                    200001          201441   6
indices = range(rows[j] - lookback, rows[j], step)   # 1440 timesteps values but in steps of 6
indices   # rrange(200001, 201441, 6)
len(indices)      # total: 240 points

# 12.2
samples[j] = float_data[indices]          # 240 points
samples[j].shape                          # 240 X 14
# 12.3
samples[j]

# 13. Our prediction datapoint is delay timesteps ahead
targets[j] = float_data[rows[j] + delay][1]   # Taregt value is the value in IInd column
                                              #  at the end of rows[j] + delay
# 13.1
targets[0]

# So finally return this sample and target
#   yield samples, targets


#############################
###### Case 3   Test data generator
#############################


#################### Expt ends ########################33


# 14. Finally use all above to define a generator to generate data:
## Our generator:
#    def generator():
#        while True:
#
#          Get 128 points from where to pick up data
#		   		For training data:
#					Select 128 random points between (0+lookback, 200000)
#				For validation/test data:
#					Start counting from i = 200001 (valid) or 300001 (test)
#					Select next 128 consecutive pts from where we finished earlier
#					i = i +128
#
#		 	Now we have 128 points either for training data or for validation/test data
#				create zero-array of size 128 X 240 X 14
#		   		Fill each one of the 128 zero-arrays with, 240 X 14 values
#          		Get an array of 128 targets at 'delay' distance
#
#          	yield sample, target
#

def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):

    if max_index is None:       # Only for test data we do not set max_index
        max_index = len(data) - delay - 1

    # 14.1 min_index = (train)0, (valid)200001, (test)300001
    i = min_index + lookback

    while 1:
        if shuffle:         #14.2 True for train data, False for valid & test data
            rows = np.random.randint(min_index + lookback,    # lower limit, min_index = 0
                                     max_index,               # Upto here, max_index = 200000
                                     size=batch_size          # Select any 128 pts
                                     )
        else:              #14.3 For validation generator rows are between (200001, 207201)
            if i + batch_size >= max_index:
                i = min_index + lookback
            #14.4 Starting i is 'min_index + lookback'
            rows = np.arange(i, min(i + batch_size, max_index))   # No random rows picking
                                                                  # Normal length of 'rows'
																  #  is batch_size or less

            #14.5 Value of this 'i' will be preserved between calls to generator()
            i += len(rows)   # Next i for valid and test is 128 distance away

        # 14.1 Create empty sample (128 X 240 X 14)
        #                      128            240         14
        samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
        # 14.2 Create 128 empty targets
        targets = np.zeros((len(rows),))

        # 14.3 For every one of the 128 points...
        for j, row in enumerate(rows):
            # 14.4 Set read-pointer to 'lookback' behind
            #      from this rows[j] and pick next 240 points
            #      at intervals of 6 points
            indices = range(rows[j] - lookback,  rows[j], step)
            # 14.5 Get data into our sample
            samples[j] = data[indices]
            # 14.6 Get targets (temperature: IINd columns).
            #      index [1] => IInd column or temperatue column
            targets[j] = data[rows[j] + delay][1]
        # 14.7 Return objects per call
        yield samples, targets


# 15. Our common constants
lookback = 1440
step = 6
delay = 144
batch_size = 128

# 16. train data generator
train_gen = generator(float_data,
                       lookback=lookback,
                       delay=delay,
                       min_index=0,
                       max_index=200000,
                       shuffle=True,
                       step=step,
                       batch_size=batch_size)

# 16.1 Have a look. Run for-loop twice to see change.
for samples, targets in train_gen:
    print(targets[0])
    print(samples[0])
    print(samples[0].shape)
    break


# 16.2
a = train_gen

# 16.2.1 Get first set of dataset
s = next(a)
type(s)            # Tuple
s[0].shape         # (128,240,1)
s[1].shape         # (128,)

# 16.2.2 Get next set of dataset
t = next(a)
t[0].shape
t[1].shape

# 17. Validation data generator
val_gen = generator(
                   float_data,
                   lookback=lookback,
                   delay=delay,
                   min_index=200001,
                   max_index= 300000,              # 207201, 300000, 200001 + 1440 *5
                   step=step,
                   batch_size=batch_size
                   )


# 18. Test data generator
test_gen = generator(
                     float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# 19 How many times to draw from validation generator
#    in order to see all validation data
val_steps = (300000 - 200001 - lookback)    # Instead of 300000-200001 - lookback to save time
val_steps        # 98559
# At the end of each epoch, validation follows.
# These many validation steps, take a long time
# for epoch to complete. As time is important
# to us, we will limit val_steps to 400, so:
# 19.1
val_steps = 400


# 20
test_steps = (len(float_data) - 300001 - lookback)
test_steps
####################################################
## 21. We will work with:
#              i) Fully connected model
#             ii) GRU model
#            iii) GRU model with dropouts
#             iv) Stacked GRU models


###################################
### AA. Fully connected model
###################################

from keras.models import Sequential
# 22. Call libraries
from keras import layers
from keras.optimizers import RMSprop

# 22.1 Model design: Sequential
model = Sequential()

# 22.2 Flatten data of one batch:
#                                         240                14
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.summary()        # 240 X 14 = 3360

# 22.3 Pass it through first hidden layer
model.add(layers.Dense(32, activation='relu'))

# 22.4 Pass it through output layer
model.add(layers.Dense(1))

# 22.5 Compile the model
model.compile(
             optimizer=RMSprop(),
             loss='mae'
             )

# 22.6
model.summary()

# 22.7 Fit/train the model
start = time.time()
history = model.fit_generator(train_gen,
                              steps_per_epoch=500, # See below for explanation
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              verbose = 1)
end = time.time()
print((end-start)/60)


# 22.8 Plot loss history
#      This function is defined at the end
plot_learning_curve()


# 22.9
#     How accuracy changes as epochs increase
def plot_learning_curve():
    val_loss = history.history['val_loss']
    tr_loss=history.history['loss']
    epochs = range(1, len(val_loss) +1)
    plt.plot(epochs,val_loss, 'b', label = "Validation MAE")
    plt.plot(epochs, tr_loss, 'r', label = "Training MAE")
    plt.title("Training and validation MAE")
    plt.legend()
    plt.show()


###################################
### BB. GRU model
###################################

# 23. Call libraries
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# 23.1 Design model
model = Sequential()

# 23.2 Add a Gated Recurrent Unit layer (similar to LSTM)
model.add(layers.GRU(
                    32,
                    input_shape=(None, float_data.shape[-1])
                    )
         )

model.summary()

# 23.3 Output layer
model.add(layers.Dense(1))

# 23.4 How does model look like?
model.summary()

# 23.5 Compile the model
model.compile(optimizer=RMSprop(), loss='mae')

# 23.6 Train the model
# Refer: https://stackoverflow.com/a/44277785
start = time.time()    # Takes 10 minutes
history = model.fit_generator(train_gen,    # Each time train_gen is called, it
                                            #  returns a sample of shape (128,240,14)
                                            #   So 240 GRU units, each receiving
                                            #    input vector of 14 features (Xt)
                                            #     Size of each batch: 240. Total batches: 128
                              steps_per_epoch=50,  # Total number of times generator called
                                                   #   & samples yielded from generator
                                                   #     Here each step brings a batch of 128
                              epochs=20,    # What is an epoch? An epoch finishes
                                           #  when steps_per_epoch batches have
                                           #    been seen by the model.
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              verbose =1)

end = time.time()
print((end-start)/60)

# 23.7 Plot loss history
#      This function is defined at the end
plot_learning_curve()


###################################
### CC. GRU model with dropouts
###################################

# Models with dropouts take longer time than without dropouts. Why? See:
#   https://stats.stackexchange.com/a/377126

# 24. Call libraries
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# 24.1 Design model
model = Sequential()

# 24.2 GRU layer with dropouts
#      What is Recurrent layer dropout?
#      See: https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout
model.add(layers.GRU(32,
                     dropout=0.2,             # Dropout of input layer
                     recurrent_dropout=0.2,   # Dropout of recurrent layer
                     input_shape=(None, float_data.shape[-1]) # float_data.shape[-1] = 14
                     )
        )

# 24.3 Output layer
model.add(layers.Dense(1))

# 24.4 Compile te model
model.compile(optimizer=RMSprop(), loss='mae')

# 24.5 Model summary
model.summary()

# 24.6 Train the model
start = time.time()     # Around 32 minutes
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              verbose =1)
end = time.time()
print((end-start)/60)

# 24.7 Plot loss history
#      This function is defined at the end
plot_learning_curve()

###################################
### DD. Stacked GRU models
###     Takes long time
###################################

# 25. Call libraries
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# 25.1 Design model
model = Sequential()
# 25.2 First GRU layer
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,    # Dropouts between two stacked GRU layers
                     return_sequences=True,    # Must for stacking another layer on top
                     input_shape=(None, float_data.shape[-1])))
# 25.3 IInd GRU layer
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))

# 25.4 Final output layer
model.add(layers.Dense(1))

# 25.5 Model summary
model.summary()

# 25.6 Compile the model
model.compile(optimizer=RMSprop(), loss='mae')

# 25.7 Train the model
start = time.time()    # Takes 82 minutes
history = model.fit_generator(train_gen,
                              steps_per_epoch=50,
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              verbose =1
                              )
end = time.time()
print((end-start)/60)

# 25.8 Plot loss history now, epoch-wise
#      This function is defined at the end
plot_learning_curve()

###################################
### EE. Bidirectional RNN
###     Takes long time
###################################
"""
What is bidirectional RNN?
    Ref: Page: 220, Book on Deep Learning by Chollet
    A bidirectional RNN is two RNNs running in parallel
    and their outputs get concatenated to be fed to
    classifier. One RNN considers input sequence as it
    comes (this is a cow). The other RNN reverses the
    sequence (cow a is this). It looks at the sequence
    from different perspective with the hope that this
    new angle may further assist in creating new features.

    As Bidirectional RNN uses two RNNs they can be GRU
    or LSTM or even SimpleRNN.

"""
# 26. Call libraries
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import time
import matplotlib.pyplot as plt

# 26.1 Develop deeplearning model using bidirectioal RNN
model = Sequential()
model.add(layers.Bidirectional(
                               layers.GRU(32),      # Can use LSTMs also
                               input_shape=(None,float_data.shape[-1])
                               )
         )

model.add(layers.Dense(1))
model.compile(
              optimizer = RMSprop(),
              loss = 'mae'
              )

start = time.time()
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 10,
                              validation_data = val_gen,
                              validation_steps=val_steps
                              )
end = time.time()
(end-start)/60

# 26.2 Plot loss history now, epoch-wise
#      This function is defined at the end
plot_learning_curve()

###########################################



"""
How to improve further?
Suggestions to further improve model:
    i)  Adjust the number of units in each recurrent layer in Stacked setup
        Current choice is arbitrary.
    ii) Adjust the learning rate used by RMSprop layer.
        Syntax: RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    iii)Try using LSTM layers instead of GRU layers. LSTM layers
        are computaionally intensive but retain more memory.
    iv) Try using a bigger densely connected regressor instead
        of simple one as used here.
    v)  Run the best performing model as per the history instead of
        the model that is returned at the end of all epochs.
    vi) Maybe add a convolution layer before recurrent layer
"""'

############ I am done #######################
