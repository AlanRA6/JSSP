import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

class JSSPSolver:
    """
    Resuelve el Job Shop Scheduling Problem usando reglas de prioridad.
    """
    
    def __init__(self, jobs: List[List[Tuple[int, int]]]):
        """
        Inicializa el solver con la instancia del problema.
        
        Args:
            jobs: Lista de trabajos. Cada trabajo es una lista de tuplas (máquina, tiempo)
                  Ejemplo: [[(0, 3), (1, 2)], [(1, 4), (0, 1)]]
        """
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = max(max(op[0] for op in job) for job in jobs) + 1
        
    def solve(self, rule: str = 'FIFO') -> Tuple[List[Dict], int]:
        """
        Resuelve el JSSP usando la regla de prioridad especificada.
        
        Args:
            rule: Regla de prioridad ('FIFO', 'SPT', 'LPT')
            
        Returns:
            Tupla con (schedule, makespan)
            schedule: Lista de operaciones programadas
            makespan: Tiempo total de finalización
        """
        # Estructuras de datos para el scheduling
        schedule = []  # Operaciones programadas

        # Arreglos de control
        machine_available = [0] * self.num_machines  # Tiempo disponible de cada máquina
        job_progress = [0] * self.num_jobs  # Próxima operación de cada trabajo
        job_available = [0] * self.num_jobs  # Tiempo disponible de cada trabajo
        
        # Total de operaciones a programar (3 + 3 + 3 = 9)
        total_operations = sum(len(job) for job in self.jobs)
        
        # Bucle principal: programar todas las operaciones
        while len(schedule) < total_operations:
            # Paso 1: Identificar operaciones disponibles
            available_ops = []
            
            for job_id in range(self.num_jobs): # J1, J2, J3
                op_index = job_progress[job_id] # Que operacion sigue?
                
                # Verificar si el trabajo tiene operaciones pendientes
                if op_index < len(self.jobs[job_id]):
                    machine, duration = self.jobs[job_id][op_index]

                    # Agregar a disponibles
                    available_ops.append({
                        'job': job_id,
                        'operation': op_index,
                        'machine': machine,
                        'duration': duration,
                        'job_ready_time': job_available[job_id]
                    })
            
            if not available_ops:
                break
            
            # Paso 2: Aplicar regla de prioridad para seleccionar operación
            selected_op = self._select_operation(available_ops, rule)
            
            # Paso 3: Calcular tiempos de inicio y fin
            # La operación inicia cuando AMBOS estén disponibles
            start_time = max(
                machine_available[selected_op['machine']],  # Máquina disponible
                job_available[selected_op['job']]           # Trabajo disponible
            )
            end_time = start_time + selected_op['duration']
            
            # Paso 4: Programar la operación
            schedule.append({
                'job': selected_op['job'],
                'operation': selected_op['operation'],
                'machine': selected_op['machine'],
                'start': start_time,
                'end': end_time,
                'duration': selected_op['duration']
            })
            
            # Paso 5: Actualizar disponibilidad
            machine_available[selected_op['machine']] = end_time
            job_available[selected_op['job']] = end_time
            job_progress[selected_op['job']] += 1
        
        # Calcular makespan (tiempo máximo de finalización)
        makespan = max(op['end'] for op in schedule)
        
        return schedule, makespan
    
    def _select_operation(self, available_ops: List[Dict], rule: str) -> Dict:
        """
        Selecciona una operación según la regla de prioridad.
        
        Args:
            available_ops: Lista de operaciones disponibles
            rule: Regla de prioridad ('FIFO', 'SPT', 'LPT')
            
        Returns:
            Operación seleccionada
        """
        if rule == 'FIFO':
            # First In First Out: seleccionar la primera en la lista
            return available_ops[0]
        
        elif rule == 'SPT':
            # Shortest Processing Time: seleccionar la de menor duración
            return min(available_ops, key=lambda op: op['duration'])
        
        elif rule == 'LPT':
            # Longest Processing Time: seleccionar la de mayor duración
            return max(available_ops, key=lambda op: op['duration'])
        
        else:
            raise ValueError(f"Regla desconocida: {rule}")
    
    def print_schedule(self, schedule: List[Dict], makespan: int, rule: str):
        """
        Imprime el schedule de forma legible.
        """
        print(f"\n{'='*60}")
        print(f"REGLA: {rule}")
        print(f"{'='*60}")
        print(f"Makespan: {makespan} unidades de tiempo\n")
        
        # Ordenar por máquina y tiempo de inicio
        schedule_sorted = sorted(schedule, key=lambda x: (x['machine'], x['start']))
        
        print(f"{'Máquina':<10} {'Trabajo':<10} {'Operación':<12} {'Inicio':<10} {'Fin':<10} {'Duración':<10}")
        print("-" * 60)
        
        for op in schedule_sorted:
            print(f"M{op['machine']:<9} J{op['job']+1:<9} Op{op['operation']+1:<11} "
                  f"{op['start']:<10} {op['end']:<10} {op['duration']:<10}")
    
    def plot_gantt(self, schedule: List[Dict], makespan: int, rule: str):
        """
        Genera un diagrama de Gantt para visualizar el schedule.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Colores para cada trabajo
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_jobs))
        
        # Dibujar cada operación
        for op in schedule:
            machine = op['machine']
            start = op['start']
            duration = op['duration']
            job = op['job']
            
            # Dibujar barra
            ax.barh(machine, duration, left=start, height=0.6, 
                   color=colors[job], edgecolor='black', linewidth=1.5)
            
            # Añadir etiqueta
            ax.text(start + duration/2, machine, f"J{job+1}\n({duration})", 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Configurar ejes
        ax.set_xlabel('Tiempo', fontsize=12, fontweight='bold')
        ax.set_ylabel('Máquina', fontsize=12, fontweight='bold')
        ax.set_title(f'Diagrama de Gantt - {rule}\nMakespan: {makespan} unidades', 
                    fontsize=14, fontweight='bold')
        
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(self.num_machines)])
        ax.set_xlim(0, makespan + 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'gantt_{rule}.png', dpi=300, bbox_inches='tight')
        print(f"Diagrama guardado como: gantt_{rule}.png")


def compare_rules(jobs: List[List[Tuple[int, int]]]):
    """
    Compara las tres reglas de prioridad y muestra resultados.
    """
    print("\n" + "="*70)
    print(" "*15 + "COMPARACIÓN DE REGLAS DE PRIORIDAD")
    print(" "*20 + "Job Shop Scheduling Problem")
    print("="*70)
    
    # Mostrar instancia
    print("\nINSTANCIA DEL PROBLEMA (3 trabajos × 3 máquinas):")
    print("-" * 70)
    for i, job in enumerate(jobs):
        operations = " → ".join([f"M{m}({t})" for m, t in job])
        print(f"Trabajo {i+1}: {operations}")
    
    # Resolver con cada regla
    rules = ['FIFO', 'SPT', 'LPT']
    results = {}
    
    solver = JSSPSolver(jobs)
    
    # Aqui se ejecuta cada regla
    for rule in rules:
        # Resolver con esta regla especifica
        schedule, makespan = solver.solve(rule) # llamada independiente

        # Guardar resultados de eesta regla
        results[rule] = {
            'schedule': schedule,
            'makespan': makespan
        }

        # Imprimir resultados de esta regla
        solver.print_schedule(schedule, makespan, rule)
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print(" "*25 + "RESUMEN COMPARATIVO")
    print(f"{'='*70}\n")
    print(f"{'Regla':<15} {'Makespan':<15} {'Diferencia vs Mejor':<20}")
    print("-" * 70)
    
    best_makespan = min(r['makespan'] for r in results.values())
    
    for rule in rules:
        makespan = results[rule]['makespan']
        diff = makespan - best_makespan
        diff_str = f"+{diff}" if diff > 0 else "ÓPTIMO"
        print(f"{rule:<15} {makespan:<15} {diff_str:<20}")
    
    # Identificar mejor regla
    best_rule = min(results, key=lambda r: results[r]['makespan'])
    print(f"\n MEJOR REGLA: {best_rule} con makespan = {results[best_rule]['makespan']}")
    
    # Generar gráficas
    print(f"\n{'='*70}")
    print("Generando diagramas de Gantt...")
    print(f"{'='*70}")
    
    for rule in rules:
        solver.plot_gantt(results[rule]['schedule'], results[rule]['makespan'], rule)
    
    # Gráfica comparativa de makespans
    plt.figure(figsize=(10, 6))
    makespans = [results[rule]['makespan'] for rule in rules]
    colors_bar = ['#3498db' if ms == min(makespans) else '#95a5a6' for ms in makespans]
    
    bars = plt.bar(rules, makespans, color=colors_bar, edgecolor='black', linewidth=2)
    plt.xlabel('Regla de Prioridad', fontsize=12, fontweight='bold')
    plt.ylabel('Makespan (unidades de tiempo)', fontsize=12, fontweight='bold')
    plt.title('Comparación de Makespan por Regla de Prioridad', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir valores en las barras
    for bar, makespan in zip(bars, makespans):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{makespan}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('comparacion_makespans.png', dpi=300, bbox_inches='tight')
    print("Gráfica comparativa guardada como: comparacion_makespans.png")
    
    return results


# ========== PROGRAMA PRINCIPAL ==========

if __name__ == "__main__":
    # Definir instancia 3×3
    # Formato: [(máquina, tiempo), (máquina, tiempo), ...]
    
    jobs = [
        [(0, 3), (1, 2), (2, 2)],  # Trabajo 1: M0(3) → M1(2) → M2(2)
        [(0, 2), (2, 1), (1, 4)],  # Trabajo 2: M0(2) → M2(1) → M1(4)
        [(1, 4), (2, 3), (0, 2)]   # Trabajo 3: M1(4) → M2(3) → M0(2)
    ]
    
    # Ejecutar comparación
    results = compare_rules(jobs)
    
    print("\n" + "="*70)
    print("Simulación completada exitosamente")
    print("="*70)
    
    # Mostrar gráficas (comentar para que no se abran automáticamente)
    #plt.show()