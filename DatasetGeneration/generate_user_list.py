import uuid
import os
import argparse


ASSET_DIR = "assets"
USER_DIR = "users"


def get_new_user(num: int) -> str:
    for _ in range(num):
        tmp_id = uuid.uuid4().time_low
        yield "user"+str(tmp_id)


def generate_user_list(file_name: str, users_num: int):
    user_list = list(set(get_new_user(users_num)))
    while len(user_list) != users_num:
        to_do = users_num - len(user_list)
        tmp = list(set(get_new_user(to_do)))
        user_list.extend(tmp)
        user_list = list(set(user_list))

    with open(os.path.join(USER_DIR, file_name), 'w') as fp:
        fp.writelines("\n".join(user_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for users list generation")
    parser.add_argument("--name", type=str, default='user_list',
                        help="User list file name")
    parser.add_argument("--num", type=str, default='5000',
                        help="Number of users")
    args = parser.parse_args()

    user_file = args.name
    number_of_users = int(args.num)
    generate_user_list(user_file, number_of_users)