import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_experiment(name="test3-11")
experiments = client.list_experiments() # returns a list of mlflow.entities.Experiment
experiments[0]._experiment_id

client.list_run_infos()

mlflow.get_run("dfa2334fe6ee4a1abca59077f9f25744").data.tags.runNmae

run = client.create_run(experiments[0].experiment_id) # returns mlflow.entities.Run
client.log_param(run.info.run_id, "hello", "world")
client.set_terminated(run.info.run_id)


mlflow.source.gi