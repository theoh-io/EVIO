import mlflow
import os
import torch
import argparse



def load_model(prev_runid, model, device):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        return model

    model_dir = run.info.artifact_uri + "/model/data/model.pth"
    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device)
        model.load_state_dict(model_loaded.state_dict())
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at" + prev_runid + "\n")

    return model

def create_model_dir(path_results, runid):
    path_results += runid + "/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def save_model(model):
    mlflow.pytorch.log_model(model, "model")

def save_state_dict(optimizer,scheduler,epoch):
    state_dict = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        # "scaler": scaler.state_dict() if scaler else None,
    }
    mlflow.pytorch.log_state_dict(state_dict, artifact_path="training_state_dict")



def log_config(path_results, runid, config):
    """
    Log configuration file to MlFlow run.
    """

    eval_id = 0
    for file in os.listdir(path_results):
        if file.endswith(".yml"):
            tmp = int(file.split(".")[0].split("_")[-1])
            eval_id = tmp + 1 if tmp + 1 > eval_id else eval_id
    yaml_filename = path_results + "eval_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()

    return eval_id


def log_results(runid, results, path, eval_id):
    """
    Log validation results as artifacts to MlFlow run.
    """
    yaml_filename = path + "metrics_" + str(eval_id) + ".yml"
    with open(yaml_filename, "w") as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    mlflow.start_run(runid)
    mlflow.log_artifact(yaml_filename)
    mlflow.end_run()

def train(args, config):
    ########## configs ##########


    # log config

    mlflow.set_tracking_uri(args.path_mlflow)
    mlflow.set_experiment(config["experiment"])  #experiment number (default 0)
    mlflow.start_run()
    mlflow.log_params(config) #log the config files
    mlflow.log_param("prev_runid", args.prev_runid)
    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])


    ########## data loader ##########

    train_dataloader = ...
    valid_dataloader = ...


    ############## Training ###############
    # model initialization and settings
    #Transformer config
    model = ...
    model.to(device)
    model = load_model(args.prev_runid, model, device)


    # Define the loss function


    # training loop
    epoch_initial = 0
    best_loss = 1000
    for epoch in range(epoch_initial, config["loader"]["n_epochs"]):
        print(f'Epoch {epoch}')
        model.train()
        for input in tqdm(train_dataloader):
            ...
  



        epoch_loss = ...
       
        if use_ml_flow:
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # save model
        with torch.no_grad():
            if epoch_loss < best_loss:
                save_model(model)
                save_state_dict(optimizer, scheduler, epoch)
                best_loss = epoch_loss


        #####validate after each 5 epoch############
        # Validation Dataset

        if epoch % config["test"]["n_valid"] == 0:

            model.eval()
            epoch_loss_valid = 0.
            print('Validating... (test sequence)')



            for input in tqdm(valid_dataloader):

                with torch.no_grad():
                    ...

            epoch_loss_valid = ...
            print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
            mlflow.log_metric("valid_loss", epoch_loss_valid, step=epoch)


    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--save_path",
        default="results/checkpoint_epoch{}.pth",
        help="save the model",
    )



