file (GLOB INSOMNIA_SOURCE 
	"*/*.cpp"
)

add_library(libInSoMniA ${INSOMNIA_SOURCE} ${INSOMNIA_HEADER})
