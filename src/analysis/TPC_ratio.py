import math
import sys
import os
import re
import argparse
import glob
import pandas as pd


def extract_ca_coords(pdb_path):
    coords = []
    processed_residues = set()

    if not os.path.exists(pdb_path):
        return None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line.split()[2] == 'CA':
                chain_id = line[21].strip()
                res_num = line[22:27].strip()
                residue_key = (chain_id, res_num)

                if residue_key not in processed_residues:
                    coords.append(line[30:54])
                    processed_residues.add(residue_key)
    return coords


def parse_residue_groups(inter_index_path):
    inter_res, no_inter_res = [], []
    if not os.path.exists(inter_index_path):
        return None, None

    with open(inter_index_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            res_idx, flag = parts[0], parts[1]
            try:
                idx = int(res_idx) - 1
                if flag == '1':
                    inter_res.append(idx)
                else:
                    no_inter_res.append(idx)
            except ValueError:
                continue
    return inter_res, no_inter_res


def _calc_dist_sq(coord_str1, coord_str2):
    x1, y1, z1 = map(float, [coord_str1[0:8], coord_str1[8:16], coord_str1[16:24]])
    x2, y2, z2 = map(float, [coord_str2[0:8], coord_str2[8:16], coord_str2[16:24]])
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2


def calculate_lddt(residue_vec, native_coords, target_coords, restrict_j=True):
    if not native_coords or not target_coords:
        return 0.0
    if len(native_coords) != len(target_coords):
        return 0.0

    global_lddt = 0.0
    valid_residues = 0
    n_coords = len(native_coords)

    for i in residue_vec:
        if i < 0 or i >= n_coords:
            continue

        cut1 = cut2 = cut3 = cut4 = 0
        num_15_total = 1e-9
        j_range = residue_vec if restrict_j else range(n_coords)

        for j in j_range:
            if j < 0 or j >= n_coords or i == j:
                continue

            native_dist_sq = _calc_dist_sq(native_coords[i], native_coords[j])
            if native_dist_sq > 225:
                continue

            native_dist = math.sqrt(native_dist_sq)
            num_15_total += 1

            target_dist = math.sqrt(_calc_dist_sq(target_coords[i], target_coords[j]))
            diff = abs(native_dist - target_dist)

            if diff <= 0.5: cut1 += 1
            if diff <= 1:   cut2 += 1
            if diff <= 2:   cut3 += 1
            if diff <= 4:   cut4 += 1

        global_lddt += (cut1 + cut2 + cut3 + cut4) / (num_15_total * 4)
        valid_residues += 1

    return global_lddt / valid_residues if valid_residues > 0 else 0.0


def process_single_protein(protein_name, model_dir, native_path, index_path, csv_output_dir):
    native_coords = extract_ca_coords(native_path)
    if not native_coords:
        print(f"[{protein_name}] Error: Cannot read Native: {native_path}")
        return None

    inter_res, no_inter_res = parse_residue_groups(index_path)
    if inter_res is None:
        print(f"[{protein_name}] Error: Cannot read Index: {index_path}")
        return None

    full_res = list(range(len(native_coords)))

    model_files = glob.glob(os.path.join(model_dir, "*.pdb"))
    if not model_files:
        print(f"[{protein_name}] Warning: No .pdb files found in -> {model_dir}")
        return None

    data_list = []

    for m_path in model_files:
        basename = os.path.basename(m_path)

        match = re.search(r'model_(\d+)\.pdb', basename)
        if match:
            model_num = int(match.group(1))
        else:
            num_search = re.search(r'(\d+)', basename)
            if num_search:
                model_num = int(num_search.group(1))
            else:
                continue

        target_coords = extract_ca_coords(m_path)
        if not target_coords or len(target_coords) != len(native_coords):
            continue

        val_inter = round(calculate_lddt(inter_res, native_coords, target_coords, restrict_j=True), 3)
        val_no_inter = round(calculate_lddt(no_inter_res, native_coords, target_coords, restrict_j=False), 3)
        val_full = round(calculate_lddt(full_res, native_coords, target_coords, restrict_j=False), 3)

        data_list.append({
            'model_number': model_num,
            'pdb_filename': basename,
            'inter_lddt': val_inter,
            'no_inter_lddt': val_no_inter,
            'full_lddt': val_full
        })

    if not data_list:
        print(f"[{protein_name}] No valid models processed (length mismatch or naming issue)")
        return None

    df = pd.DataFrame(data_list)
    df = df.sort_values('model_number')

    if csv_output_dir:
        if csv_output_dir.endswith('.csv'):
            csv_output_dir = os.path.dirname(csv_output_dir)
        os.makedirs(csv_output_dir, exist_ok=True)

        out_df = df[['pdb_filename', 'inter_lddt', 'no_inter_lddt', 'full_lddt']].copy()
        out_df.columns = ['pdb_filename', 'EFR_lddt', 'LFR_lddt', 'full_lddt']

        csv_save_path = os.path.join(csv_output_dir, f"{protein_name}.csv")
        out_df.to_csv(csv_save_path, index=False)
        print(f"[{protein_name}] Detailed CSV generated: {csv_save_path}")

    total_valid = len(df)
    exclude_count = 50

    sorted_nums = df['model_number'].unique()
    if len(sorted_nums) > 2 * exclude_count:
        start_num = sorted_nums[exclude_count]
        end_num = sorted_nums[-(exclude_count + 1)]
        df_filtered = df[(df['model_number'] >= start_num) & (df['model_number'] <= end_num)]
    else:
        df_filtered = df

    filtered_count = len(df_filtered)
    advantage_count = ((df_filtered['inter_lddt'] - df_filtered['no_inter_lddt']) >= 0.1).sum()
    ratio = advantage_count / filtered_count if filtered_count > 0 else 0.0
    last_full_lddt = df['full_lddt'].iloc[-1] if not df.empty else 0.0

    return {
        'total': total_valid,
        'filtered': filtered_count,
        'advantage': advantage_count,
        'ratio': ratio,
        'last_lddt': last_full_lddt
    }


def main():
    parser = argparse.ArgumentParser(description="LDDT Calculation Tool (Single File & Batch Directory Modes)")

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing models or root directory with protein folders')
    parser.add_argument('--native_dir', required=True, help='Native PDB path (File or Directory)')
    parser.add_argument('--index_dir', required=True, help='Index file path (File or Directory)')
    parser.add_argument('--summary_output', required=True, help='Summary output path (.csv or .tsv)')
    parser.add_argument('--csv_dir', required=True, help='Directory to save detailed CSVs')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)

    real_csv_dir = args.csv_dir
    if args.csv_dir.endswith('.csv'):
        real_csv_dir = os.path.dirname(args.csv_dir)
        if not real_csv_dir: real_csv_dir = '.'

    header = "Protein\tTotal\tFiltered\tAdvantage\tRatio\tFull_LDDT\n"
    if args.summary_output.endswith('.csv'):
        header = "Protein,Total,Filtered,Advantage,Ratio,Full_LDDT\n"
        sep = ","
    else:
        sep = "\t"

    with open(args.summary_output, 'w', encoding='utf-8') as f_out:
        f_out.write(header)

        if os.path.isfile(args.native_dir) and os.path.isfile(args.index_dir):
            protein_name = os.path.splitext(os.path.basename(args.native_dir))[0]

            result = process_single_protein(
                protein_name,
                args.input_dir,
                args.native_dir,
                args.index_dir,
                real_csv_dir
            )

            if result:
                line = f"{protein_name}{sep}{result['total']}{sep}{result['filtered']}{sep}" \
                       f"{result['advantage']}{sep}{result['ratio']:.4f}{sep}{result['last_lddt']:.3f}\n"
                f_out.write(line)
                print(f"Processing complete! Summary saved to: {args.summary_output}")

        else:
            if not os.path.exists(args.input_dir):
                sys.exit(f"Error: Input directory does not exist {args.input_dir}")

            subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
            subdirs.sort()
            print(f"Found {len(subdirs)} subdirectories to process.")

            for protein_name in subdirs:
                model_dir = os.path.join(args.input_dir, protein_name)
                native_path = os.path.join(args.native_dir, protein_name + ".pdb")
                index_path = os.path.join(args.index_dir, protein_name + ".txt")

                result = process_single_protein(protein_name, model_dir, native_path, index_path, real_csv_dir)

                if result:
                    line = f"{protein_name}{sep}{result['total']}{sep}{result['filtered']}{sep}" \
                           f"{result['advantage']}{sep}{result['ratio']:.4f}{sep}{result['last_lddt']:.3f}\n"
                    f_out.write(line)
                    f_out.flush()
                    print(f"Finished: {protein_name}")


if __name__ == "__main__":
    main()