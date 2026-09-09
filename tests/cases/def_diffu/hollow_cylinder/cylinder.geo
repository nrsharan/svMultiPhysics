SetFactory("OpenCASCADE");
Ri = 1.0;
Ro = 1.3;
L = 2.0;
lc = 0.35;

Circle(1) = {0,0,0, Ro};
Circle(2) = {0,0,0, Ri};
Curve Loop(1) = {1};
Curve Loop(2) = {2};
Plane Surface(1) = {1, 2};

out[] = Extrude {0,0,L} { Surface{1}; };

Physical Surface("bottom") = {1};
Physical Surface("top") = {out[0]};
Physical Surface("outer") = {out[2]};
Physical Surface("inner") = {out[3]};
Physical Volume("tube") = {out[1]};

Mesh.CharacteristicLengthMax = lc;
Mesh.CharacteristicLengthMin = lc*0.6;
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 1;
