# Geometries

These geometries are created using [cardiac-geometries](https://computationalphysiology.github.io/cardiac_geometries)


The slab geometry is created using the following command
```bash
cardiac-geometries create-slab slab --lx=5.0 --ly=2.0 --lz=1.0 --dx=1.0 --create-fibers --fiber-angle-epi=0 --fiber-angle-endo=0
```

The LV geometry was created using the following command
```bash
cardiac-geometries create-lv-ellipsoid lv --create-fibers --psize-ref=7 --fiber-space="Quadrature_3"
```
