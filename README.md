# [kkBox DataGame](https://www.kaggle.com/competitions/datagame-2023/overview)
## How to Download the Data?
1. `pip install --user kaggle`
2. get the `kaggle.json` from kaggle:
    1. Open the top left your photo, and click the **Your Profile**.
    2. Then, click the **Account**, and click **Create New Token** at the API.
    3. It will automatically down load `kaggle.json` to your device.
3. run `mkdir ~/.kaggle`
4. put your `kaggle.json` to `~/.kaggle`, and run `chmod 600 ~/.kaggle/kaggle.json`
5. run `kaggle competitions download -c datagame-2023` in your terminal, and the data will be download to your current folder. (Put to `/data`)
## What is in the Folders?
1. data: data download from kaggle
2. notbook: code 
3. submittion: submit data to the leaderboard
## Current Process
- Finished coding encoder-decoder-version, not include all the feactures. (11/21, 14:31)
- Now training the encoder-decoder model.
## To Do
- Use all the feature for training.
- Try another model.
## Bug
- training process last two batch dimension error (fixed)


