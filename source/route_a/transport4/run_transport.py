"""Finite-volume current spreading through ideal, explicitly specified contacts."""
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent


def save(name, value):
    path = HERE / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def contact_mask(nx, fraction, patches):
    assert nx % patches == 0
    pitch = nx // patches
    count = int(round(pitch * fraction))
    assert abs(count - pitch * fraction) < 1e-12
    one = np.zeros(pitch, dtype=bool)
    start = (pitch - count) // 2
    one[start:start+count] = True
    edges = np.zeros(pitch, dtype=bool)
    if fraction < 1 and count:
        assert count % 4 == 0
        edges[start:start+count//4] = True
        edges[start+3*count//4:start+count] = True
    return np.tile(one, patches), np.tile(edges, patches)


@lru_cache(maxsize=4)
def base_operator(nx, ny, width):
    dx, dy = width / nx, 1. / ny
    x = sparse.diags([-np.ones(nx-1), 2*np.ones(nx), -np.ones(nx-1)], [-1, 0, 1], format='lil')
    x[0, nx-1] = -1
    x[nx-1, 0] = -1
    diagonal = 2*np.ones(ny)
    diagonal[[0, -1]] = 1
    y = sparse.diags([-np.ones(ny-1), diagonal, -np.ones(ny-1)], [-1, 0, 1], format='csc')
    return (sparse.kron(sparse.eye(ny), x.tocsc()*(dy/dx), format='csc')
            + sparse.kron(y*(dx/dy), sparse.eye(nx), format='csc'))


def solve(nx, ny, width, mask, *, conductivity=1., current=1., bottom_value=None):
    if not np.any(mask):
        return {'status': 'disconnected', 'reason': 'No connected sink for imposed current.'}, None
    dx, dy = width / nx, 1. / ny
    gb = 2*conductivity*dx/dy
    boundary = np.zeros(nx*ny)
    boundary[:nx] = gb*mask
    operator = conductivity*base_operator(nx, ny, width) + sparse.diags(boundary, format='csc')
    rhs = np.zeros(nx*ny)
    rhs[-nx:] = current*dx
    vb = np.zeros(nx) if bottom_value is None else np.asarray(bottom_value)
    rhs[:nx] += gb*mask*vb
    values = spsolve(operator, rhs)
    potential = values.reshape(ny, nx)
    face_current = np.where(mask, 2*conductivity*(potential[0]-vb)/dy, 0.)
    top = potential[-1] + current*dy/(2*conductivity)
    incoming, outgoing = current*width, float(np.sum(face_current)*dx)
    horizontal = conductivity*dy/dx*np.sum((potential - np.roll(potential, -1, axis=1))**2)
    vertical = conductivity*dx/dy*np.sum(np.diff(potential, axis=0)**2)
    bottom_power = gb*np.sum(mask*(potential[0]-vb)**2)
    top_power = current**2*width*dy/(2*conductivity)
    dissipation = float(horizontal + vertical + bottom_power + top_power)
    work = float(current*dx*np.sum(top) - dx*np.sum(face_current*vb))
    residual = operator @ values - rhs
    result = {
        'status': 'solved', 'nx': nx, 'ny': ny, 'width_over_height': width,
        'conductivity': conductivity, 'applied_current': current,
        'contact_fraction': float(np.mean(mask)),
        'contact_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'effective_resistance': float(np.mean(top)/current),
        'resistance_over_full_contact': float(np.mean(top)/current*conductivity),
        'bottom_current_integral': outgoing, 'top_current_integral': incoming,
        'current_relative_error': abs(outgoing-incoming)/abs(incoming),
        'zero_gap_current': bool(np.all(face_current[~mask] == 0)),
        'active_current_mean_over_applied': float(np.mean(face_current[mask])/current),
        'maximum_face_current_over_applied': float(np.max(face_current)/current),
        'dissipation': dissipation, 'boundary_work': work,
        'power_relative_error': abs(dissipation-work)/max(abs(work), 1e-30),
        'linear_residual_relative_l2': float(np.linalg.norm(residual)/np.linalg.norm(rhs)),
    }
    return result, (potential, face_current)


def consistency_checks():
    checkpoint_guard()
    full = []
    for conductivity in [.3, 1., 3.]:
        for current in [.2, 1., 4.]:
            mask = np.ones(32, dtype=bool)
            result, fields = solve(32, 32, 1., mask, conductivity=conductivity, current=current)
            y = (np.arange(32)+.5)/32
            error = float(np.max(abs(fields[0] - current*y[:, None]/conductivity)))
            result['maximum_field_error'] = error
            result['exact_resistance'] = 1/conductivity
            result['pass'] = bool(error <= 1e-9*(1+current/conductivity)
                                  and abs(result['effective_resistance']-1/conductivity) <= 1e-9*(1+1/conductivity))
            full.append(result)
    disconnected, disconnected_fields = solve(32, 32, 1., np.zeros(32, dtype=bool))
    mms = []
    for n in [16, 32, 64, 128]:
        checkpoint_guard()
        x, y = (np.arange(n)+.5)/n, (np.arange(n)+.5)/n
        vb = .03*np.cos(2*np.pi*x)
        result, fields = solve(n, n, 1., np.ones(n, dtype=bool), bottom_value=vb)
        exact = y[:, None] + .03*np.cos(2*np.pi*x)[None, :]*np.cosh(2*np.pi*(1-y[:, None]))/np.cosh(2*np.pi)
        result['rms_field_error'] = float(np.sqrt(np.mean((fields[0]-exact)**2)))
        if mms:
            result['observed_order'] = float(np.log2(mms[-1]['rms_field_error']/result['rms_field_error']))
        mms.append(result)
    mms_pass = mms[-1]['rms_field_error'] <= 1e-4 and all(x['observed_order'] >= 1.8 for x in mms[-2:])
    all_rows = full + mms
    conservation_pass = all(x['current_relative_error'] <= 1e-9 and x['power_relative_error'] <= 1e-9
                            and x['linear_residual_relative_l2'] <= 1e-9 for x in all_rows)
    result = {'full_contact': full, 'manufactured_solution': mms, 'disconnected': disconnected,
              'full_contact_pass': all(x['pass'] for x in full), 'mms_pass': bool(mms_pass),
              'conservation_pass': conservation_pass,
              'all_pass': all(x['pass'] for x in full) and mms_pass and conservation_pass and disconnected_fields is None}
    # The no-contact return was explicitly checked without attempting a singular solve.
    result['all_pass'] = bool(result['all_pass'] and disconnected['status'] == 'disconnected')
    save('consistency_checks.json', result)
    return result


def geometry_cases():
    cases = [{'name': 'full', 'width': 1., 'fraction': 1., 'patches': 1}]
    for fraction in [.75, .5, .25]:
        for patches in [1, 4]:
            cases.append({'name': f'area{fraction:g}_patches{patches}', 'width': 1., 'fraction': fraction, 'patches': patches})
    for width in [.25, 4.]:
        cases.append({'name': f'area0.5_width{width:g}', 'width': width, 'fraction': .5, 'patches': 1})
    return cases


def summarize(cases):
    pairs = []
    for geometry in geometry_cases():
        group = sorted([r for r in cases if r['name'] == geometry['name']], key=lambda r: r['ny'])
        for a, b in zip(group, group[1:]):
            resistance_change = abs(a['effective_resistance'] - b['effective_resistance'])/abs(b['effective_resistance'])
            edge_change = abs(a['edge_current_share'] - b['edge_current_share'])
            pairs.append({'name': geometry['name'], 'coarse_ny': a['ny'], 'fine_ny': b['ny'],
                          'resistance_relative_change': resistance_change,
                          'edge_share_absolute_change': edge_change,
                          'resistance_pass': resistance_change <= .01,
                          'edge_share_pass': edge_change <= .01})
    monotonic = []
    for ny in sorted({r['ny'] for r in cases}):
        for patches in [1, 4]:
            group = sorted([r for r in cases if r['ny'] == ny and r['width_over_height'] == 1
                            and (r['patches'] == patches or r['contact_fraction'] == 1)],
                           key=lambda r: -r['contact_fraction'])
            monotonic.append({'ny': ny, 'patches': patches,
                              'pass': all(b['effective_resistance'] >= a['effective_resistance']-1e-9 for a, b in zip(group, group[1:]))})
    finest_pairs = [r for r in pairs if r['fine_ny'] == 256]
    result = {'cases': len(cases), 'comparisons': pairs, 'rayleigh_monotonicity': monotonic,
              'all_conservation_pass': all(r['current_relative_error'] <= 1e-9 and r['power_relative_error'] <= 1e-9
                                           and r['linear_residual_relative_l2'] <= 1e-9 and r['zero_gap_current'] for r in cases),
              'finest_comparison_count': len(finest_pairs),
              'finest_resistance_pass_count': sum(r['resistance_pass'] for r in finest_pairs),
              'finest_edge_share_pass_count': sum(r['edge_share_pass'] for r in finest_pairs),
              'maximum_face_current_is_converged_physical_peak': False,
              'cathode_amplification_predicted': False}
    save('summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = {'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), HERE/'PLAN.md']},
                'numpy': np.__version__, 'scipy': scipy.__version__, 'geometry': geometry_cases()}
    if args.resume:
        assert json.loads((HERE/'manifest.json').read_text()) == manifest, 'Changed manifest; do not mix cases.'
        cases = json.loads((HERE/'cases.json').read_text())
        checks = json.loads((HERE/'consistency_checks.json').read_text())
    else:
        assert not (HERE/'cases.json').exists(), 'Preserve existing evidence; use --resume or a separate copy.'
        save('manifest.json', manifest)
        cases = []
        checks = consistency_checks()
    if not checks['all_pass']:
        print(json.dumps({'status': 'consistency_failure', 'checks': checks['all_pass']}))
        return
    for ny in ([64] if args.pilot else [64, 128, 256]):
        for geometry in geometry_cases():
            if any(r['name'] == geometry['name'] and r['ny'] == ny for r in cases):
                continue
            checkpoint_guard()
            tick = time.perf_counter()
            nx = int(round(geometry['width']*ny))
            mask, edge = contact_mask(nx, geometry['fraction'], geometry['patches'])
            result, fields = solve(nx, ny, geometry['width'], mask)
            result.update({'name': geometry['name'], 'patches': geometry['patches'],
                           'edge_current_share': float(np.sum(fields[1][edge])/np.sum(fields[1])),
                           'elapsed_seconds': time.perf_counter()-tick})
            if ny == 256:
                result['surface_profile'] = {'x_over_height': ((np.arange(nx)+.5)*geometry['width']/nx).tolist(),
                                             'contact': mask.astype(int).tolist(),
                                             'edge_region': edge.astype(int).tolist(),
                                             'current_over_applied': fields[1].tolist()}
            cases.append(result)
            save('cases.json', cases)
            print(json.dumps({k: result[k] for k in ['name', 'ny', 'effective_resistance', 'edge_current_share', 'elapsed_seconds']}), flush=True)
            del fields
    summary = summarize(cases)
    print(json.dumps({'cases': len(cases), 'all_conservation_pass': summary['all_conservation_pass'],
                      'finest_qualified_resistance': summary['finest_resistance_pass_count'],
                      'finest_qualified_edge_share': summary['finest_edge_share_pass_count']}))


if __name__ == '__main__':
    main()
