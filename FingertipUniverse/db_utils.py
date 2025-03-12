import logging

from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def mysql_engine(host, port, user, passwd, db=None):
    """
    生成mysql数据库链接
    :param host:
    :param port:
    :param user:
    :param passwd:
    :param db:
    :return:
    """
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}')
        return engine
    except Exception as e:
        logger.error(f"An error occurred: {e}")