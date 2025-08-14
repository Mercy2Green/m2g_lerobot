# cd ~/vla/src/pyorbbecsdk
# export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
# sudo bash ./scripts/install_udev_rules.sh
# sudo udevadm control --reload-rules
# sudo udevadm trigger
# cd /home/hpx/peter_ws/m2g_lerobot/lerobot


# conda activate ur
cd /home/hpx/peter_ws/m2g_lerobot/lerobot
DATA_SAVE_DIR="m2g"
# rm -r /media/hpx/newMemery/datasets/${DATA_SAVE_DIR}

export PYTHONPATH="/home/hpx/peter_ws/m2g_lerobot:$PYTHONPATH"

# ===== Task definitions =====
task_1="Please grasp the empty plastic water bottle, lift it, and put it down."        # 请抓起塑料水瓶，抬起并放下。
task_2="Please grasp the roll of tape, lift it, and put it down."          # 请抓起大卷胶带，抬起并放下。
task_3="Please grasp the black toy car wheel, lift it, and put it down."         # 请抓起黑色玩具车轮，抬起并放下。
task_4="Please grasp the boxed beverage, lift it, and put it down."              # 请抓起盒装饮料，抬起并放下。
task_5="Please grasp the mechanical gripper, lift it, and put it down."          # 请抓起机械夹爪，抬起并放下。
task_6="Please grasp the plastic toy chicken, lift it, and put it down."         # 请抓起塑料玩具鸡，抬起并放下。
task_7="Please grasp the clay cylinder, lift it, and put it down."               # 请抓起圆柱形橡皮泥，抬起并放下。

# ===== Choose which task to run =====
TASK_NAME="task_6"       # 这里改任务名，例如 task_2, task_3 ...
CURRENT_TASK=${!TASK_NAME}  # 取变量值
# ===== Root folder =====
ROOT_DIR="/home/hpx/peter_ws/data/grasp/${DATA_SAVE_DIR}"  # 数据保存的根目录
SAVE_PATH="${ROOT_DIR}/${TASK_NAME}"  # 保存到 task_X 文件夹

###### For debug
# rm -r $SAVE_PATH
################

# rm -r SAVE_PATH
# ===== Run data collection =====
python /home/hpx/peter_ws/m2g_lerobot/lerobot/record_m2g.py \
  --srobot.type=koch_follower \
  --srobot.port=/ttyUSB0 \
  --teleop.type=koch_leader \
  --teleop.port=/ttyUSB1 \
  --dataset.repo_id=m2g/"$CURRENT_TASK" \
  --dataset.root="$SAVE_PATH" \
  --dataset.push_to_hub=False \
  --dataset.num_episodes=20 \
  --dataset.single_task="$CURRENT_TASK" \
  --auto_mode=True \
  --force=80 \
  --down_height=0.145 \
  --grasp_pose="[400, 400, 400, 400, 600, 0]"


