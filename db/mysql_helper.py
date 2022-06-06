from db.db_utils_init import get_my_connection

"""
# 封装好的数据库工具
# 执行select有结果返回结果,没有返回0；
# 增/删/改返回变更数据条数，没有返回0
"""


class MySqlHelper(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'inst'): # 单例
            cls.inst = super(MySqlHelper, cls).__new__(cls, *args, **kwargs)
        return cls.inst

    def __init__(self):
        # 初始化数据库连接池
        self.db = get_my_connection()

    # 封装execute命令 执行后返回从连接池获取的cursor和conn
    def execute(self, sql, param=None, autoclose=False):
        """
        主要判断是否有参数和是否执行完就释放连接
        :param sql: 字符串类型，sql语句
        :param param: sql语句中要替换的参数"select %s from tab where id=%s" 其中的%s就是参数 元组或列表形式
        :param autoclose: 执行sql后是否自动关闭连接
        :return: 返回连接conn和游标cursor, 以及 count
        """
        # 从连接池获取连接
        cursor, conn = self.db.getconn()
        # count : 改变的数据条数
        count = 0
        try:
            if param:
                count = cursor.execute(sql, param)
            else:
                count = cursor.execute(sql)
            conn.commit()
            if autoclose:
                self.close(cursor, conn)
        except Exception as e:
            print(e)
        return cursor, conn, count

    # 释放连接，归还给连接池
    def close(self, cursor, conn):
        cursor.close()
        conn.close()

    # 查询所有 返回数据元组
    def select_all(self, sql, param=None):
        try:
            cursor, conn, count = self.execute(sql, param)
            res = cursor.fetchall()
            return res
        except Exception as e:
            print(e)
            self.close(cursor, conn)
            return count

    # 查询单条
    def select_one(self, sql, param=None):
        try:
            cursor, conn, count = self.execute(sql, param)
            res = cursor.fetchone()
            self.close(cursor, conn)
            return res
        except Exception as e:
            print("error_msg:", e.args)
            self.close(cursor, conn)
            return count

    # 插入单条
    def insert_one(self, sql, param):
        try:
            cursor, conn, count = self.execute(sql, param)
            conn.commit()
            self.close(cursor, conn)
            return count
        except Exception as e:
            print(e)
            conn.rollback()
            self.close(cursor, conn)
            return count

    # 删除
    def delete(self, sql, param=None):
        try:
            cursor, conn, count = self.execute(sql, param)
            self.close(cursor, conn)
            return count
        except Exception as e:
            print(e)
            conn.rollback()
            self.close(cursor, conn)
            return count

    # 更新
    def update(self, sql, param=None):
        try:
            cursor, conn, count = self.execute(sql, param)
            conn.commit()
            self.close(cursor, conn)
            return count
        except Exception as e:
            print(e)
            conn.rollback()
            self.close(cursor, conn)
            return count

