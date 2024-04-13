import json
import time

class Filter:
    def log(self, msg):
        # print('[{}] {}'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))
        with open('./filter.log', 'a', encoding='utf-8') as f:
            f.write('[INFO][{}] {}\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))

if __name__ == "__main__":
    f = Filter()
    print(f)