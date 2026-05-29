from datetime import datetime, timedelta, timezone

def current_date():
    # 获取当前 UTC 时间
    now_utc = datetime.utcnow()
    # 定义东八区的时区偏移量
    utc_plus_8 = timezone(timedelta(hours=8))
    # 将当前时间转换为东八区时间
    now_eight = now_utc.replace(tzinfo=timezone.utc).astimezone(utc_plus_8)
    # 格式化为字符串
    time_str = now_eight.strftime('%Y-%m-%d %H:%M:%S')
    return time_str


def simple_dateformat(datetime_obj):
    return datetime_obj.strftime('%Y-%m-%d %H:%M:%S')


def get_now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_now_str_code():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def current_date_str():
    # 获取当前 UTC 时间
    now_utc = datetime.utcnow()
    # 定义东八区的时区偏移量
    utc_plus_8 = timezone(timedelta(hours=8))
    # 将当前时间转换为东八区时间
    now_eight = now_utc.replace(tzinfo=timezone.utc).astimezone(utc_plus_8)
    # 格式化为字符串
    time_str = now_eight.strftime('%Y-%m-%d')
    return time_str


if __name__ == '__main__' :
    print(f'当前东八区时间: {current_date()}')
