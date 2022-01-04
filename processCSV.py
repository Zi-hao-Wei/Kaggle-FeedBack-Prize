import pandas as pd
train_df = pd.read_csv('corrected_train.csv')
train_df = train_df[["id","discourse_id","discourse_start","discourse_end", "discourse_type", "text_by_index", "new_start",
                     "new_end", "text_by_new_index", "new_predictionstring"]]
print(train_df.head())

for t, df in list(train_df.groupby("discourse_type", as_index=False)):
    print(t, df.shape)
train_df.to_csv("corrected.csv",index=False)
