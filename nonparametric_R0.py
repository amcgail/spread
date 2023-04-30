from .common import *

def estimate_R0(serial_intervals, debug=False):
    """
    Estimate R0 from serial intervals using nonparametric method.
    Call this function with a list, one item for each infection.
    For each pair where A infects B, the entry should include the number of days between A's infection and B's infection.
    """
    import subprocess

    with open('tmp.csv', 'w') as outf:
        outf.write('\n'.join([
            f'{y:0.6f}' for x,y in serial_interval.items()
        ]))

    cmd = 'Rscript estimate_R0.R tmp.csv'
    process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode

    import csv
    ret = out.decode('utf8')
    if debug:
        print(ret)
        print('\n\n\n\n')

    r0_est = csv.DictReader( csv.StringIO(ret) )
    r0_est = list(r0_est)
    for x in r0_est:
        for y in x:
            x[y] = float(x[y])
    r0_est = pd.DataFrame.from_records(r0_est)