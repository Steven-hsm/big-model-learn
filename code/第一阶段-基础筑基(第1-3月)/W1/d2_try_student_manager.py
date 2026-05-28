def add_student(students, student_data):
    sid = student_data['id']
    if sid in students:
        raise ValueError(f"学生ID {sid} 已存在")
    students[sid] = student_data.copy()
    return students[sid]

def find_student(students, student_id):
    return students.get(student_id, None)

def update_student(students, student_id, **kwargs):
    if student_id not in students:
        raise KeyError(f"学生ID {student_id} 不存在")
    students[student_id].update(kwargs)
    return students[student_id]

def delete_student(students, student_id):
    if student_id in students:
        return students.pop(student_id)
    return None

def list_students(students, sort_by='name'):
    return sorted(students.values(), key=lambda s: s.get(sort_by, ''))

def get_statistics(students):
    all_scores = [s for stu in students.values() for s in stu.get('scores', [])]
    return {
        'count': len(students),
        'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'max_score': max(all_scores) if all_scores else 0,
        'min_score': min(all_scores) if all_scores else 0,
    }

# 测试
if __name__ == '__main__':
    students = {}
    add_student(students, {'id': 1, 'name': '张三', 'age': 22, 'scores': [85, 90], 'grade': 'A'})
    add_student(students, {'id': 2, 'name': '李四', 'age': 23, 'scores': [78, 82], 'grade': 'B'})
    print(list_students(students))
    print(get_statistics(students))