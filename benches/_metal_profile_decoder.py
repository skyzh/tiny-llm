"""Headless access to an existing Metal profile with installed Apple code.

Version-specific local diagnostic adapter, not a supported public Apple API.
No window, new capture, GPU dispatch, buffer export or package installation.
"""
import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

FRAMEWORK = '/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/Versions/A/GTShaderProfiler'
C.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation', mode=C.RTLD_GLOBAL)
framework = C.CDLL(FRAMEWORK, mode=C.RTLD_GLOBAL)
objc = C.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes = [C.c_char_p]
objc.objc_getClass.restype = C.c_void_p
objc.sel_registerName.argtypes = [C.c_char_p]
objc.sel_registerName.restype = C.c_void_p
objc.object_getClassName.argtypes = [C.c_void_p]
objc.object_getClassName.restype = C.c_char_p
objc.class_copyMethodList.argtypes = [C.c_void_p, C.POINTER(C.c_uint)]
objc.class_copyMethodList.restype = C.POINTER(C.c_void_p)
objc.method_getName.argtypes = [C.c_void_p]
objc.method_getName.restype = C.c_void_p
objc.sel_getName.argtypes = [C.c_void_p]
objc.sel_getName.restype = C.c_char_p
objc.method_getTypeEncoding.argtypes = [C.c_void_p]
objc.method_getTypeEncoding.restype = C.c_char_p
objc.object_getClass.argtypes = [C.c_void_p]
objc.object_getClass.restype = C.c_void_p
libc = C.CDLL('/usr/lib/libSystem.B.dylib')
libc.free.argtypes = [C.c_void_p]
address = C.cast(objc.objc_msgSend, C.c_void_p).value

def cls(name):
    ptr = objc.objc_getClass(name.encode())
    assert ptr, name
    return ptr

def call(obj, selector, result=C.c_void_p, args=()):
    assert obj, selector
    func = C.CFUNCTYPE(result, C.c_void_p, C.c_void_p, *(t for t, _ in args))(address)
    return func(obj, objc.sel_registerName(selector.encode()), *(v for _, v in args))

def string(text):
    return call(cls('NSString'), 'stringWithUTF8String:', args=((C.c_char_p, text.encode()),))

def text(obj):
    if not obj:
        return None
    return call(obj, 'UTF8String', C.c_char_p).decode()

def cname(obj):
    return objc.object_getClassName(obj).decode() if obj else None

def methods(obj):
    n = C.c_uint()
    ptr = objc.class_copyMethodList(objc.object_getClass(obj), C.byref(n))
    result = []
    for i in range(n.value):
        result.append({'selector': objc.sel_getName(objc.method_getName(ptr[i])).decode(),
                       'encoding': objc.method_getTypeEncoding(ptr[i]).decode()})
    libc.free(ptr)
    return result

def safe_summary(obj):
    if not obj:
        return None
    return {'class': cname(obj), 'methods': methods(obj)}

class CostContext(C.Structure):
    _fields_ = [('level', C.c_ushort), ('scope', C.c_ushort),
                ('level_identifier', C.c_uint), ('scope_identifier', C.c_uint64)]

class CostInfo(C.Structure):
    # Field order is pinned by the installed Objective-C type encoding.
    # Unnamed channels remain unnamed until independently reconciled.
    _fields_ = [('context', CostContext), ('value0', C.c_double),
                ('channels0', C.c_double * 10), ('value1', C.c_double),
                ('channels1', C.c_double * 10), ('count0', C.c_uint64),
                ('counts', C.c_uint64 * 10), ('tail', C.c_uint64 * 3)]

class DebugLocation(C.Structure):
    _fields_ = [('function_index', C.c_uint), ('path_index', C.c_uint),
                ('line', C.c_uint), ('column', C.c_uint)]

class DebugRange(C.Structure):
    _pack_ = 4
    _fields_ = [('instruction_index', C.c_uint), ('address', C.c_uint64),
                ('length', C.c_uint64), ('location_index', C.c_uint),
                ('parent_range_index', C.c_uint)]

class BuilderOptions(C.Structure):
    _fields_ = [('filter_by_command_buffer_times', C.c_bool),
                ('generate_standalone_config', C.c_bool),
                ('use_legacy_LLVM_processing', C.c_bool),
                ('use_legacy_clique_processing', C.c_bool)]

def extract(mio):
    assert C.sizeof(CostInfo) == 304 and C.sizeof(DebugLocation) == 16
    array = call(mio, 'binaries')
    n = call(array, 'count', C.c_uint64)
    assert n < 256
    gpu = CostInfo.from_buffer_copy(call(mio, 'gpuCost', C.POINTER(CostInfo)).contents)
    result = {'gpu_value0': gpu.value0, 'gpu_value1': gpu.value1,
              'gpu_channels0': list(gpu.channels0), 'gpu_channels1': list(gpu.channels1),
              'gpu_instruction_count': gpu.count0,
              'gpu_context': [gpu.context.level, gpu.context.scope,
                              gpu.context.level_identifier, gpu.context.scope_identifier],
              'binaries': []}
    result['trace_counts'] = {name: call(mio, name, C.c_uint64)
                              for name in ('drawCount', 'costCount', 'kicksCount', 'drawTraceCount', 'totalCliqueCost')}
    for i in range(n):
        binary = call(array, 'objectAtIndex:', args=((C.c_uint64, i),))
        count = call(binary, 'debugLocationCount', C.c_uint64)
        assert count < 100000
        own = CostInfo.from_buffer_copy(call(binary, 'cost', C.POINTER(CostInfo)).contents)
        api_total = call(binary, 'totalCostForScope:scopeIdentifier:dataMaster:', C.c_double,
                         args=((C.c_ushort, own.context.scope), (C.c_uint64, own.context.scope_identifier),
                               (C.c_ushort, 2)))
        locations = call(binary, 'debugLocations', C.POINTER(DebugLocation))
        rows = []
        sample_locations = []
        unique = set()
        for j in range(count):
            loc = locations[j]
            if j < 8:
                sample_locations.append({'raw': [loc.function_index, loc.path_index, loc.line, loc.column],
                                         'file': text(call(binary, 'debugFilePathForDebugLocationAtIndex:', args=((C.c_uint, j),))),
                                         'function': text(call(binary, 'debugFunctionNameForDebugLocationAtIndex:', args=((C.c_uint, j),)))})
            if not loc.line or (loc.path_index, loc.line) in unique:
                continue
            unique.add((loc.path_index, loc.line))
            fullpath = text(call(binary, 'debugFilePathForDebugLocationAtIndex:', args=((C.c_uint, j),)))
            function = text(call(binary, 'debugFunctionNameForDebugLocationAtIndex:', args=((C.c_uint, j),)))
            mapped_path_index = call(binary, 'fullPathIndexForFilePath:', C.c_uint,
                                     args=((C.c_void_p, string(fullpath)),))
            assert mapped_path_index != 0xffffffff
            cost = CostInfo()
            instructions = C.c_uint()
            ok = call(binary, 'costForLine:fullPathIndex:scope:scopeIdentifier:cost:numInstructions:', C.c_bool,
                      args=((C.c_uint, loc.line), (C.c_uint, mapped_path_index),
                            (C.c_ushort, own.context.scope),
                            (C.c_uint64, own.context.scope_identifier),
                            (C.POINTER(CostInfo), C.byref(cost)),
                            (C.POINTER(C.c_uint), C.byref(instructions))))
            if ok:
                rows.append({'file': fullpath, 'function': function, 'line': loc.line,
                             'path_index': loc.path_index, 'value0': cost.value0,
                             'value1': cost.value1, 'channels0': list(cost.channels0),
                             'num_instructions': instructions.value})
        leaf = {}
        instruction_count = call(binary, 'instructionInfoCount', C.c_uint64)
        assert instruction_count < 100000 and C.sizeof(DebugRange) == 28
        summed = 0.0
        unmapped = 0.0
        for k in range(instruction_count):
            cost = CostInfo()
            ok = call(binary, 'instructionCost:scope:scopeIdentifier:cost:', C.c_bool,
                      args=((C.c_uint, k), (C.c_ushort, own.context.scope),
                            (C.c_uint64, own.context.scope_identifier),
                            (C.POINTER(CostInfo), C.byref(cost))))
            value = cost.value0 + cost.value1
            if not ok or not value:
                continue
            summed += value
            pointer = call(binary, 'debugRangeForInstructionAtIndex:', C.POINTER(DebugRange), args=((C.c_uint, k),))
            if not pointer or pointer.contents.location_index >= count:
                unmapped += value
                continue
            index = pointer.contents.location_index
            loc = locations[index]
            file = text(call(binary, 'debugFilePathForDebugLocationAtIndex:', args=((C.c_uint, index),)))
            function = text(call(binary, 'debugFunctionNameForDebugLocationAtIndex:', args=((C.c_uint, index),)))
            key = (file, function, loc.line)
            entry = leaf.setdefault(key, {'file': file, 'function': function, 'line': loc.line,
                                          'alu_cost': 0.0, 'non_alu_cost': 0.0, 'instruction_sites': 0})
            entry['alu_cost'] += cost.value0
            entry['non_alu_cost'] += cost.value1
            entry['instruction_sites'] += 1
        assert abs(summed - (own.value0 + own.value1)) < 1e-7, 'INSTRUCTION_COST_TOTAL_MISMATCH'
        assert abs(sum(x['alu_cost'] + x['non_alu_cost'] for x in leaf.values()) + unmapped - summed) < 1e-7
        result['binaries'].append({'index': i, 'class': cname(binary),
                                   'debug_location_count': count,
                                   'value0': own.value0, 'value1': own.value1,
                                   'API_compute_total': api_total,
                                   'static_instruction_count': instruction_count,
                                   'instruction_cost_sum': summed, 'unmapped_cost': unmapped,
                                   'leaf_rows': list(leaf.values()),
                                   'channels0': list(own.channels0), 'channels1': list(own.channels1),
                                   'instruction_count': own.count0, 'tail': list(own.tail),
                                   'sample_locations': sample_locations,
                                   'context': [own.context.level, own.context.scope,
                                               own.context.level_identifier, own.context.scope_identifier],
                                   'rows': rows})
    return result

def pipeline_timings(result):
    if not result:
        return []
    shader_result = call(result, 'shaderProfilerResult')
    pipelines = call(shader_result, 'pipelineStates')
    def array_values(value):
        if call(value, 'isKindOfClass:', C.c_bool, args=((C.c_void_p, cls('NSDictionary')),)):
            return call(value, 'allValues')
        assert call(value, 'isKindOfClass:', C.c_bool, args=((C.c_void_p, cls('NSArray')),))
        return value
    values = array_values(pipelines)
    n = call(values, 'count', C.c_uint64)
    assert n < 256
    rows = []
    for i in range(n):
        pipeline = call(values, 'objectAtIndex:', args=((C.c_uint64, i),))
        timing = call(pipeline, 'timingInfo')
        functions = call(pipeline, 'shaderFunctions')
        function_values = array_values(functions)
        nf = call(function_values, 'count', C.c_uint64)
        assert nf < 32
        names = [text(call(call(function_values, 'objectAtIndex:', args=((C.c_uint64, j),)), 'name')) for j in range(nf)]
        rows.append({'object_id': call(pipeline, 'objectId', C.c_uint64),
                     'shader_functions': names,
                     'time_ns': call(timing, 'time', C.c_uint64),
                     'compute_time_ns': call(timing, 'computeTime', C.c_uint64)})
    return rows

def main():
    pool = call(cls('NSAutoreleasePool'), 'new')
    run = Path(os.environ['FORGE_RUN'])
    source = Path(sys.argv[1])
    assert source.name == 'streamData' and source.stat().st_size < 4 * 1024**2
    # The saved APS metadata names its existing raw profiling files relatively.
    # Preserve the archive and resolve those references from their own directory.
    os.chdir(source.parent)
    # The directory form of Apple's loader binds dataFileURL, resolving the
    # existing Profiling_f_N.raw references. The plain-file form does not.
    url = call(cls('NSURL'), 'fileURLWithPath:', args=((C.c_void_p, string(str(source.parent))),))
    stream = call(cls('GTShaderProfilerStreamData'), 'dataFromArchivedDataURL:', args=((C.c_void_p, url),))
    assert stream, 'STREAM_DECODE_NIL'
    helper = Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/GPUToolsPlatform/PlugIns/GTLLVMHelper')
    helper_value = string(str(helper)) if helper.is_file() else None
    direct = len(sys.argv) > 2 and sys.argv[2].startswith('direct')
    if direct:
        options = BuilderOptions(sys.argv[2] == 'direct-filter', False, False, False)
        processor, result = None, None
        mio = call(cls('GTMioTraceData'), 'alloc')
        mio = call(mio, 'initWithStreamData:llvmHelperPath:options:',
                   args=((C.c_void_p, stream), (C.c_void_p, helper_value), (BuilderOptions, options)))
        call(mio, 'loadTimeline', None)
    else:
        processor = call(cls('GTShaderProfilerStreamDataProcessor'), 'alloc')
        processor = call(processor, 'initWithStreamData:llvmHelperPath:', args=((C.c_void_p, stream), (C.c_void_p, helper_value)))
        assert processor
        call(processor, 'processStreamData', None)
        call(processor, 'waitUntilFinished', None)
        result = call(processor, 'result')
        mio = call(processor, 'mioData')
    print('LOAD_COST_TIMELINE_BEGIN', flush=True)
    call(mio, 'loadCostTimeline', None)
    deadline = time.monotonic() + 12
    while call(mio, 'loadingCostTimeline', C.c_bool):
        assert time.monotonic() < deadline, 'COST_TIMELINE_TIMEOUT'
        time.sleep(.01)
    print('LOAD_COST_TIMELINE_END', flush=True)
    cost_timeline = call(mio, 'costTimeline')
    print('COST_TIMELINE_CLASS', cname(cost_timeline), flush=True)
    facts = {'stream': cname(stream), 'processor': cname(processor),
             'result': safe_summary(result), 'mio': safe_summary(mio),
             'helper': str(helper) if helper_value else None,
             'has_separate_costs': call(mio, 'hasSeparateCostsTimeline', C.c_bool)}
    file_url = call(stream, 'dataFileURL')
    facts['data_file_URL'] = text(call(file_url, 'absoluteString')) if file_url else None
    facts['cost_timeline'] = safe_summary(cost_timeline)
    facts['pipeline_timings'] = pipeline_timings(result)
    if result:
        selectors = {x['selector'] for x in facts['result']['methods']}
        for key in ('mioData', 'shaderProfilerResult'):
            if key in selectors:
                value = call(result, key)
                facts[key] = safe_summary(value)
    raw = json.dumps(facts, indent=2).encode()
    assert len(raw) < 1024**2
    (run / 'decoded-profile-types.json').write_bytes(raw)
    assert cname(cost_timeline) == 'GTMioTraceTimelineData'
    extracted = extract(cost_timeline)
    payload = json.dumps(extracted, indent=2).encode()
    assert len(payload) < 2 * 1024**2
    (run / 'line-cost-data.json').write_bytes(payload)
    print(json.dumps({'gpu_weighted_cost': extracted['gpu_value0'] + extracted['gpu_value1'],
                      'binaries': [{'index': b['index'], 'cost': b['value0'] + b['value1'],
                                    'inclusive_source_rows': len(b['rows']),
                                    'instruction_sum_matches': abs(b['instruction_cost_sum'] - b['value0'] - b['value1']) < 1e-7}
                                   for b in extracted['binaries']]}))
    print(json.dumps({k: v if not isinstance(v, dict) else v.get('class') for k, v in facts.items()}))
    call(mio if direct else processor, 'release', None)
    call(pool, 'drain', None)

if __name__ == '__main__':
    main()
