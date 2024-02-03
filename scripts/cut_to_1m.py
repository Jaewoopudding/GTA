import numpy as np
import os


if __name__ == "__main__":
    dirnm = "/input"
    datas = os.listdir(f"{dirnm}/")
    for d in datas:
        if d.endswith(".npz"):

            data = np.load(f'{dirnm}/{d}', allow_pickle=True)

            gen_data = data['data']
            config = data['config'].item()

            np.random.shuffle(gen_data)
            m = (1000000//31)+1
            gen_data = gen_data[:m+1]


            np.savez(f'{dirnm}/{d}', data=gen_data, config=config)

