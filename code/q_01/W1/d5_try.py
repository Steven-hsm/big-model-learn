 simple_orm.py - 简易ORM实现
# 要求：支持 define_table / insert / find / delete 操作

class SimpleORM:
    def __init__(self):
        self._tables = {}

    def define_table(self, table_name, columns):
        """定义表结构"""
        self._tables[table_name] = {
            'columns': columns,
            'rows': [],
            'next_id': 1
        }

    def insert(self, table_name, **data):
        """插入一条记录"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        table = self._tables[table_name]
        row = {'id': table['next_id']}
        for col in table['columns']:
            row[col] = data.get(col, None)
        table['rows'].append(row)
        table['next_id'] += 1
        return row

    def find(self, table_name, **conditions):
        """按条件查询"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        rows = self._tables[table_name]['rows']
        if not conditions:
            return rows
        return [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]

    def delete(self, table_name, **conditions):
        """按条件删除"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        table = self._tables[table_name]
        original_count = len(table['rows'])
        table['rows'] = [r for r in table['rows']
                         if not all(r.get(k) == v for k, v in conditions.items())]
        return original_count - len(table['rows'])

# 测试
if __name__ == '__main__':
    db = SimpleORM()
    db.define_table('students', ['name', 'age', 'grade'])
    db.insert('students', name='张三', age=22, grade='A')
    db.insert('students', name='李四', age=23, grade='B')
    print(db.find('students', grade='A'))
    print(db.delete('students', name='张三'))
    print(db.find('students'))
