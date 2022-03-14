# ai9_demura_mlflow_tutorial


```shell
AWS_ACCESS_KEY_ID=a19_admin \
AWS_SECRET_ACCESS_KEY=a19_sample_key \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9100 \
MLFLOW_TRACKING_URI=http://localhost:5100 \
mlflow run git@github.com:tony92151/ai9_demura_mlflow_tutorial.git -e demurarun \
-P csv_dataset="/home/tedbest/datadisk/a19/repo_to_upload/ai9_mura_dataset_2022_backup2/20220210_merged_258/data_merged.csv" \
-P experiment_name="test3-14-today" \
-P parent_name="lr_test_on_resnet50"
```

```shell
AWS_ACCESS_KEY_ID=a19_admin \
AWS_SECRET_ACCESS_KEY=a19_sample_key \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9100 \
MLFLOW_TRACKING_URI=http://localhost:5100 \
mlflow run git@repo.a19.center:tony92151/ai9_demura_mlflow_tutorial.git -e run \
-P csv_dataset="/home/tedbest/datadisk/a19/repo_to_upload/ai9_mura_dataset_2022_backup2/20220210_merged_258/data_merged.csv" \
-P experiment_name="test3-14-today" \
-P parent_name="lr_test_on_resnet50"
```

