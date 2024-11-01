import csv

def sortedCsv(input_csv, output_csv):
    try:
        with open(input_csv, "r") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)

            # Mantener la última columna fija y ordenar las demás
            last_column = header[-1]
            columns_to_sort = header[:-1]
            sorted_indices = sorted(range(len(columns_to_sort)), key=lambda i: columns_to_sort[i])
            sorted_header = [columns_to_sort[i] for i in sorted_indices] + [last_column]

            # Reorganizar las filas según el orden de `sorted_indices`, manteniendo la última columna
            sorted_rows = [
                [row[i] for i in sorted_indices] + [row[-1]]
                for row in reader
            ]

        # Escribir el nuevo archivo CSV ordenado
        with open(output_csv, "w", newline="") as sorted_csv_file:
            writer = csv.writer(sorted_csv_file)
            writer.writerow(sorted_header)  # Escribir la cabecera ordenada
            writer.writerows(sorted_rows)   # Escribir las filas reordenadas

        print(f"CSV ordenado y guardado como '{output_csv}'.")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")
