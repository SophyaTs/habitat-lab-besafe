eval "$(conda shell.bash hook)"
conda activate habitat
python -u -m habitat_baselines.run --config-name=pointnav/ppo_pointnav_unreal.yaml habitat_baselines.evaluate=True

