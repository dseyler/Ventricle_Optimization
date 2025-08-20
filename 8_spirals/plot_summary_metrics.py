import os
import re
import matplotlib.pyplot as plt
import argparse

SUMMARY_PREFIX = 'summary_ventricle_'
SUMMARY_SUFFIX = '.txt'

FILENAME_REGEX = re.compile(r'summary_ventricle_\d+_(\d+)_([ce])\.txt')
TWIST_REGEX = re.compile(r'Final Twist Angle\s*:\s*([\d\.Ee+-]+)')
RATIO_REGEX = re.compile(r'Twist/Volume Ratio\s*:\s*([\d\.Ee+-]+)')
CIRC_STRAIN_REGEX = re.compile(r'Final Circumferential Strain\s*:\s*([\d\.Ee+-]+)')
LONG_STRAIN_REGEX = re.compile(r'Final Longitudinal Strain\s*:\s*([\d\.Ee+-]+)')
FINAL_VOL_REGEX = re.compile(r'Final Volume\s*:\s*([\d\.Ee+-]+)')

def extract_metrics_from_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    twist_match = TWIST_REGEX.search(content)
    ratio_match = RATIO_REGEX.search(content)
    circ_strain_match = CIRC_STRAIN_REGEX.search(content)
    long_strain_match = LONG_STRAIN_REGEX.search(content)
    final_vol_match = FINAL_VOL_REGEX.search(content)
    final_twist = float(twist_match.group(1)) if twist_match else None
    twist_vol_ratio = float(ratio_match.group(1)) if ratio_match else None
    circ_strain = float(circ_strain_match.group(1)) if circ_strain_match else None
    long_strain = float(long_strain_match.group(1)) if long_strain_match else None
    final_vol = float(final_vol_match.group(1)) if final_vol_match else None
    missing = []
    if twist_match is None:
        missing.append('Final Twist Angle')
    if ratio_match is None:
        missing.append('Twist/Volume Ratio')
    if circ_strain_match is None:
        missing.append('Final Circumferential Strain')
    if long_strain_match is None:
        missing.append('Final Longitudinal Strain')
    if final_vol_match is None:
        missing.append('Final Volume')
    return final_twist, twist_vol_ratio, circ_strain, long_strain, final_vol, missing

def main():
    parser = argparse.ArgumentParser(description='Plot Final Twist Angle, Twist/Volume Ratio, Strains, and Volume from summary files.')
    parser.add_argument('summary_dir', nargs='?', default='all_summaries_8_spirals', help='Directory containing summary_ventricle_*.txt files')
    args = parser.parse_args()
    summary_dir = os.path.abspath(args.summary_dir)
    files = [f for f in os.listdir(summary_dir) if f.startswith(SUMMARY_PREFIX) and f.endswith(SUMMARY_SUFFIX)]
    x_c, twist_c, ratio_c, circ_c, long_c, vol_c = [], [], [], [], [], []
    x_e, twist_e, ratio_e, circ_e, long_e, vol_e = [], [], [], [], [], []
    for fname in files:
        match = FILENAME_REGEX.match(fname)
        if not match:
            print(f"Skipping file (pattern mismatch): {fname}")
            continue
        x_val = int(match.group(1))
        tag = match.group(2)
        filepath = os.path.join(summary_dir, fname)
        final_twist, twist_vol_ratio, circ_strain, long_strain, final_vol, missing = extract_metrics_from_file(filepath)
        if None in (final_twist, twist_vol_ratio, circ_strain, long_strain, final_vol):
            print(f"Skipping file (missing metrics: {', '.join(missing)}): {fname}")
            continue
        if tag == 'c':
            x_c.append(x_val)
            twist_c.append(final_twist)
            ratio_c.append(twist_vol_ratio)
            circ_c.append(circ_strain)
            long_c.append(long_strain)
            vol_c.append(final_vol)
        elif tag == 'e':
            x_e.append(x_val)
            twist_e.append(final_twist)
            ratio_e.append(twist_vol_ratio)
            circ_e.append(circ_strain)
            long_e.append(long_strain)
            vol_e.append(final_vol)
    # Sort by x_vals
    def sort_group(x, *ys):
        if not x:
            return ([],) * (1 + len(ys))
        sorted_data = sorted(zip(x, *ys))
        return tuple(zip(*sorted_data))
    x_c, twist_c, ratio_c, circ_c, long_c, vol_c = sort_group(x_c, twist_c, ratio_c, circ_c, long_c, vol_c)
    x_e, twist_e, ratio_e, circ_e, long_e, vol_e = sort_group(x_e, twist_e, ratio_e, circ_e, long_e, vol_e)
    # Plot Final Twist Angle
    plt.figure()
    if x_c:
        plt.plot(x_c, twist_c, 'o-b', label='Cartesian')
    if x_e:
        plt.plot(x_e, twist_e, 'o-r', label='Ellipsoidal')
    plt.xlabel('Helix Angle')
    plt.ylabel('Final Twist Angle')
    plt.title('Final Twist Angle vs Helix Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'final_twist_angle_vs_param.png'), dpi=300)
    # Plot Twist/Volume Ratio
    plt.figure()
    if x_c:
        plt.plot(x_c, ratio_c, 'o-b', label='Cartesian')
    if x_e:
        plt.plot(x_e, ratio_e, 'o-r', label='Ellipsoidal')
    plt.xlabel('Helix Angle')
    plt.ylabel('Twist/Volume Ratio')
    plt.title('Twist/Volume Ratio vs Helix Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'twist_volume_ratio_vs_param.png'), dpi=300)
    # Plot Final Strains (both on same plot)
    plt.figure()
    if x_c:
        plt.plot(x_c, circ_c, 's-b', label='Circumferential Strain')
        plt.plot(x_c, long_c, 'd-b', label='Longitudinal Strain')
    if x_e:
        plt.plot(x_e, circ_e, 's-r', label='Circumferential Strain')
        plt.plot(x_e, long_e, 'd-r', label='Longitudinal Strain')
    plt.xlabel('Helix Angle')
    plt.ylabel('Strain')
    plt.title('Final Strains vs Helix Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'final_strains_vs_param.png'), dpi=300)
    # Plot Final Volume
    plt.figure()
    if x_c:
        plt.plot(x_c, vol_c, 'o-b', label='Cartesian')
    if x_e:
        plt.plot(x_e, vol_e, 'o-r', label='Ellipsoidal')
    plt.xlabel('Helix Angle')
    plt.ylabel('Final Volume')
    plt.title('Final Volume vs Helix Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'final_volume_vs_param.png'), dpi=300)
    print('Plots saved in', summary_dir)

if __name__ == '__main__':
    main() 