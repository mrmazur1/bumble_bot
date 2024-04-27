# bumble-bot

Bot to automate my personal Bumble account. It uses a custom-made Pytorch-trained Neural Network model to calculate the attractiveness/compatibility that I would have with a user account. After calculations, it either likes or dislikes the account and can repeat this process automatically for hours. It can detect faces/people in photos and can react to various Bumble pop-ups/ads/messages to continue matching.

# VERY IMPORTANT READ BELOW

this is a work in progress and does not have a gui, Also the .pth file that holds all the values for the model weights is on a google drive (which you need for the program to run). This is because I dont feel like paying for more storage. link is below.

https://drive.google.com/file/d/1vGT0zNjq2gkdoMsERkVfuecAPoC8v2yu/view?usp=sharing

# Training

I am working on a GUI, but you can train your own model in the 'training' branch. you have to do some coding, but it just involves you inputing custom values in the .pth model you are training. I have some comments explaining it in main.py. you can train with cpu but it takes forever so id only train if you have a GPU.

# steps
1: run pip install -r requirements. You need about 2.5 GB for all the packages. I recommend using an Anaconda environment or a virtual one. 

2: you need to download the .pth file if you dont want to train your own. The training branch has code for training and you can customize your 'hot' and 'not' folder for your preferences. This is under NN_data/images/ (i need to change the name). To be warned, training takes a while (hours). Otherwise, download the .pth from the google drive link.

3: I am working on making selenium work with chrome/firefox, but Edge was the easiest to get working initially somehow and it needs the most up to date edge driver. The project currently has one in the 'web driver' folder but it may be out of date one day. so get it if you need to.

4: now you should be able to run the code normally. If this is your first time, select an arbitrary amount for the 'number of swipes' input and then 1 for the 'what do you want to do section'. the program will run and wait until you tell it to continue. before continuing, log in successfully with your phone or email and then run. you need to do this before bumble bot saves your login info. 

5: just let it run. and leave it alone. it will look like its not doing anything but it takes a sec or two to analyze photos. Also ignore the warnings its just for deprecated code.


