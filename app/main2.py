# Definir los trabajos y mostrar la estructura

# Cada trabajp es una lista de operaciones (maquina, duracion)

jobs = [
    [(0, 3), (1, 2), (2, 2)], # Trabajo 1
    [(0, 2), (2, 1), (1, 4)], # Trabajo 2
    [(1, 4), (2, 3), (0, 2)], # Trabajo 3
]

# Mostrar los trabajos
print("INSTANCIAS DEL PROBLEMA")
print("=======================\n")

for i, job in enumerate(jobs):
    operaciones = " -> ".join([f"M{m}({t})" for m, t in job])
    print(f"Trabajo {i+1}: {operaciones}")


# Simulacion basica sin reglas

num_jobs = len(jobs)
num_machines = max(max(op[0] for op in job) for job in jobs) + 1

# Disponibilidad de cada maquina y trabajo
machine_available = [0] * num_machines
job_available = [0] * num_jobs

# Para guardar el resultado del schedule
schedule = []

print("\n\nEJECUCIÓN DE OPERACIONES (orden natural de los trabajos):")
print("==========================================================")

# Recorrer las operaciones en orden
for job_id, job in enumerate(jobs):
    for op_index, (machine, duration) in enumerate(job):
        start_time = max(machine_available[machine], job_available[job_id])
        end_time = start_time + duration

        # Guardar datos
        schedule.append({
            'job': job_id + 1,
            'operation': op_index + 1,
            'machine': machine,
            'start': start_time,
            'end': end_time,
            'duration': duration
        })


        # Actualizar disponibilidad
        machine_available[machine] = end_time
        job_available[job_id] = end_time

        print(f"Trabajo {job_id + 1}, Operacion {op_index+1} -> "
              f"M{machine} | Inicio: {start_time}, Fin: {end_time}")


# Calcular makespan
makespan = max(op['end'] for op in schedule)

print("\nTiempo total (makespan):", makespan)