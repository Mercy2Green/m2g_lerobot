
python -m lerobot.record_m2g \
  --srobot.type=koch_follower \
  --srobot.port=/ttyUSB0 \
  --teleop.type=koch_leader \
  --teleop.port=/ttyUSB1 \
  --dataset.repo_id=hpx/cube1 \
  --dataset.root=/media/hpx/newMemery/datasets/cube8 \
  --dataset.push_to_hub=False \
  --dataset.num_episodes=100 \
  --dataset.single_task="Grab the toy and put it in the box"
