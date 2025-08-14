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
task_1="Please pinch the hard black toy car wheel, lift it, and put it down."  # 请捏起硬质黑色玩具车轮，抬起并放下。
task_2="Please pinch the hard pink foam cube, lift it, and put it down."       # 请捏起硬质粉色泡沫方块，抬起并放下。
task_3="Please pinch the soft eggplant plush toy, lift it, and put it down."   # 请捏起软质茄子玩偶，抬起并放下。
task_4="Please pinch the ultra-soft tomato toy, lift it, and put it down."     # 请捏起超软西红柿玩具，抬起并放下。
task_5="Please pinch the plastic chick, lift it, and put it down."             # 请捏起塑料小鸡，抬起并放下。
task_6="Please pinch the soft pink silicone cup, lift it, and put it down."    # 请捏起软质粉色硅胶杯子，抬起并放下。
task_7="Please pinch the soft clay cube, lift it, and put it down."            # 请捏起软质橡皮泥方块，抬起并放下。
task_8="Please pinch the soft clay cylinder, lift it, and put it down."        # 请捏起软质橡皮泥圆柱，抬起并放下。
task_9="Please pinch the plastic medicine bottle, lift it, and put it down."   # 请捏起塑料药瓶，抬起并放下。
task_10="Please pinch the mouse, lift it, and put it down."                    # 请捏起鼠标，抬起并放下。

# ===== Choose which task to run =====
TASK_NAME="task_7"       # 这里改任务名，例如 task_2, task_3 ...
CURRENT_TASK=${!TASK_NAME}  # 取变量值
# ===== Root folder =====
ROOT_DIR="/home/hpx/peter_ws/data/pinch/${DATA_SAVE_DIR}"  # 数据保存的根目录
SAVE_PATH="${ROOT_DIR}/${TASK_NAME}"  # 保存到 task_X 文件夹
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
  --tall_flag=True \
  --soft_flag=True \
  --auto_mode=True


