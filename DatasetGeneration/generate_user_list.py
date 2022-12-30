import uuid
import os


NUM_USER = 1000
ASSET_DIR = "assets"


def generate_user_list(num: int) -> str:
    for _ in range(num):
        tmp = uuid.uuid4().time_low
        yield "user"+str(tmp)


if __name__ == "__main__":
    user_list = list(set(generate_user_list(NUM_USER)))
    while len(user_list) != NUM_USER:
        to_do = NUM_USER - len(user_list)
        tmp = list(set(generate_user_list(to_do)))
        user_list.extend(tmp)
        user_list = list(set(user_list))

    with open(os.path.join(ASSET_DIR, "user_list"), 'w') as fp:
        fp.writelines("\n".join(user_list))
