CC      = clang
CFLAGS  ?= -O3 -march=native
LDLIBS  = -lm -lpthread

cmxc: cmxc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: cmxc
	sh tests/roundtrip.sh ./cmxc

clean:
	rm -f cmxc

.PHONY: test clean
