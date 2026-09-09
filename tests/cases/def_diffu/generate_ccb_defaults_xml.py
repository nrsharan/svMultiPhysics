#!/usr/bin/env python3
"""Generate a <CCBActiveCMMGandR> XML block from the AceGen kernel's own
compiled-in default domain-data values.

Interface2's C++ wrapper (AceGenInterface::DeformationDiffusionConstrained
MixtureModelSmoothMuscleActiveGrowthReorientationTetrahedra3D10) does not
expose these defaults through its public API -- getDomainData() returns
null for the introspection-only constructors, even though the underlying
AceGen kernel computes them internally (SMTSetElSpecCMMSMCActiveGrowthReorientation
sets es->Data = defd when called with ng == -1; the wrapper just never
copies that into a member the caller can read). So this script reads them
directly out of the generated C source's 'gdcs' (names) and 'defd' (default
values) arrays.

Name cleaning mirrors ace_gen_cmm_smc_element.cpp's clean_domain_data_name()
EXACTLY, so the <Name> tags this script emits are guaranteed to match what
svMultiPhysics will actually look up at runtime -- see the long comment in
that function for why the raw names need cleaning at all (AceGen's naming
convention for this kernel is inconsistent: most entries are
"<math-form> -<CleanName>" but at least one is "<CleanName> -<description
with spaces>").

Usage:
    python3 generate_ccb_defaults_xml.py [path/to/SMC_Interface_Active_Growth_CMM_Reorientation.c]

Defaults to the copy this repository was integrated against.
"""

import re
import sys

DEFAULT_SOURCE = (
    "/Users/sramesh/dev/interface/Interface2/src/AceGen/"
    "SMC_Interface_Active_Growth_CMM_Reorientation.c"
)


def clean_domain_data_name(raw: str) -> str:
    """Mirror ace_gen_cmm_smc_element.cpp's clean_domain_data_name()."""
    dash = raw.find("-")
    if dash == -1:
        candidate = raw
    else:
        underscore = raw.find("_", dash)
        candidate = raw[dash + 1 :] if underscore == -1 else raw[dash + 1 : underscore]

    looks_invalid = (not candidate) or (" " in candidate)
    if looks_invalid and dash != -1:
        candidate = raw[:dash].rstrip()

    return candidate


def extract_array(text: str, array_name: str) -> str:
    """Return the raw '{...}' body of `static <type> <array_name>[]={...};`."""
    pattern = re.compile(
        r"static\s+(?:char\s*\*|double)\s*" + re.escape(array_name) + r"\s*\[\]\s*=\s*\{",
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not find array '{array_name}' in source.")

    # Walk forward from the opening brace, tracking nesting and string
    # literals, to find the matching closing brace.
    start = m.end()
    depth = 1
    i = start
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 1  # skip escaped char
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
        i += 1
    raise ValueError(f"Unterminated array '{array_name}' in source.")


def parse_string_array(body: str):
    return re.findall(r'"((?:[^"\\]|\\.)*)"', body)


def parse_double_array(body: str):
    # Strip whitespace/newlines then split on top-level commas (no nested
    # structures in this array, so a plain split is safe).
    tokens = [t.strip() for t in body.split(",")]
    tokens = [t for t in tokens if t != ""]
    values = []
    for t in tokens:
        # C-style 'e0' exponent suffix (e.g. "30e0") parses fine as a float
        # in Python since 'e0' is a valid exponent marker.
        values.append(float(t))
    return values


def main():
    source_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    with open(source_path, "r") as f:
        text = f.read()

    names_raw = parse_string_array(extract_array(text, "gdcs"))
    values = parse_double_array(extract_array(text, "defd"))

    no_domain_data_match = re.search(r"es->id\.NoDomainData\s*=\s*(\d+)", text)
    if not no_domain_data_match:
        raise ValueError("Could not find es->id.NoDomainData in source.")
    n = int(no_domain_data_match.group(1))

    if len(names_raw) != n:
        raise ValueError(f"gdcs has {len(names_raw)} entries but NoDomainData={n}.")
    if len(values) < n:
        raise ValueError(f"defd has only {len(values)} entries but NoDomainData={n}.")
    if len(values) != n:
        sys.stderr.write(
            f"Note: defd has {len(values)} entries but only the first {n} "
            "(NoDomainData) are used by the kernel; ignoring the rest.\n"
        )

    print("<CCBActiveCMMGandR>")
    for raw_name, value in zip(names_raw, values[:n]):
        name = clean_domain_data_name(raw_name)
        print(f"  <{name}> {value:.10g} </{name}>")
    print("</CCBActiveCMMGandR>")


if __name__ == "__main__":
    main()
