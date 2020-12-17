import subprocess


def pull(branch_name):
    """
    When on a Colab Notebook, resets the current branch and pulls some other branch
    :param branch_name: Name of the branch to pull from
    """
    execute_and_get_output('git fetch')
    execute_and_get_output('git add -A')
    execute_and_get_output('git reset --hard')
    execute_and_get_output('git checkout {}'.format(branch_name))
    execute_and_get_output('git pull')
    execute_and_get_output('git branch')
    execute_and_get_output('ls')


def execute_and_get_output(command):
    """
    Executes a bash command, and prints the output
    :param command: Bash command to be executed
    """
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('utf-8')))


def download_model(model_name, save_dir, files):
    """
    When on Colab Notebook, downloads a trained model on your local filesystem
    :param model_name: Name of the model to be saved
    :param save_dir: Directory in which to save the model
    :param files: Imported in Colab, allows access to the Colab filesystem
    """
    zip_cmd = 'zip -r ./{}.zip ./{}'.format(model_name, save_dir)
    execute_and_get_output(zip_cmd)
    files.download('./{}.zip'.format(model_name))

# Add these lines to the notebook to pull your branch on the Drive
# if COLAB:
#     from google.colab import drive
#     drive.mount('/content/drive')
#     drive_path = '/content/drive/Shareddrives/ML_Road_Segmentation/CS-433-project-2-{name}/project_road_segmentation'
#     os.chdir(drive_path)
#     from helpers.colab import mount_and_pull
#     BRANCH_NAME = 'name_of_your_branch'
#     mount_and_pull(BRANCH_NAME)
#
# And these lines after saving your model to download it on your local system
# (MODEL_NAME and the save directory should be set in the Notebook)
# if COLAB:
#     from helpers.colab import download_model
#     from google.colab import files
#     download_model(MODEL_NAME, SAVE_DIR, files)

