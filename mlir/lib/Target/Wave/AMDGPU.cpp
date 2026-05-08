//===- AMDGPU.cpp - WaveMachine to AMDGPU backend -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Wave/AMDGPU.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/Dialect/Wave/IR/WaveAMD.h"
#include "mlir/Dialect/Wave/Transforms/Passes.h"
#include "mlir/Dialect/WaveMachine/IR/WaveMachine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/Targets.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>

using namespace mlir;

namespace {

static bool isWM(Operation *op) {
  return op->getName().getDialectNamespace() ==
         wavemachine::WaveMachineDialect::getDialectNamespace();
}

struct KernelArgInfo {
  std::string name;
  unsigned offset = 0;
  unsigned size = 0;
  bool isGlobalBuffer = false;
};

struct KernelInfo {
  std::string name;
  unsigned kernargSize = 0;
  unsigned sgprCount = 0;
  unsigned vgprCount = 0;
  SmallVector<KernelArgInfo> args;
};

class WaveAMDGPUEmitter {
public:
  explicit WaveAMDGPUEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emit(Operation *op) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitError("wave AMDGPU backend expects a module operation");
    if (failed(initializeMC(op)))
      return failure();

    os << "\t.text\n";
    os << "\t.amdgcn_target \"amdgcn-amd-amdhsa--gfx1100\"\n";
    os << "\t.amdhsa_code_object_version 6\n";
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(emitFunction(func)))
        return failure();
    }
    emitMetadata();
    return success();
  }

private:
  raw_ostream &os;
  std::unique_ptr<llvm::MCRegisterInfo> mri;
  std::unique_ptr<llvm::MCAsmInfo> mai;
  std::unique_ptr<llvm::MCInstrInfo> mcii;
  std::unique_ptr<llvm::MCSubtargetInfo> sti;
  std::unique_ptr<llvm::MCContext> mcContext;
  std::unique_ptr<llvm::MCInstPrinter> instPrinter;
  SmallVector<KernelInfo> kernels;
  unsigned indent = 1;

  LogicalResult initializeMC(Operation *op) {
    static llvm::once_flag initializeBackendOnce;
    llvm::call_once(initializeBackendOnce, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmPrinters();
    });
    llvm::Triple triple("amdgcn-amd-amdhsa");
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
      return op->emitError("failed to lookup AMDGPU target: ") << error;
    llvm::MCTargetOptions mcOptions;
    mri.reset(target->createMCRegInfo(triple));
    mai.reset(target->createMCAsmInfo(*mri, triple, mcOptions));
    mcii.reset(target->createMCInstrInfo());
    sti.reset(target->createMCSubtargetInfo(triple, "gfx1100", ""));
    mcContext = std::make_unique<llvm::MCContext>(
        triple, *mai, mri.get(), sti.get());
    unsigned asmVariant = mai->getOutputAssemblerDialect();
    instPrinter.reset(target->createMCInstPrinter(triple, asmVariant, *mai,
                                                 *mcii, *mri));
    if (!instPrinter)
      return op->emitError("failed to create AMDGPU MCInstPrinter");
    return success();
  }

  void emitLine(StringRef line) {
    for (unsigned i = 0; i < indent; ++i)
      os << '\t';
    os << line << '\n';
  }

  void emitLine(const Twine &line) {
    SmallString<128> storage;
    emitLine(line.toStringRef(storage));
  }

  unsigned getIntAttr(Operation *op, StringRef name, unsigned fallback) const {
    if (auto attr = op->getAttrOfType<IntegerAttr>(name))
      return attr.getInt();
    return fallback;
  }

  bool isBufferPointer(Type type) const {
    auto ptr = dyn_cast<wave::PtrType>(type);
    return ptr && isa<waveamd::BufferAddressSpaceAttr>(ptr.getAddressSpace());
  }

  unsigned kernelArgSize(Type type) const {
    auto ptr = dyn_cast<wave::PtrType>(type);
    if (!ptr)
      return 4;
    return isBufferPointer(type) ? 16 : 8;
  }

  LogicalResult emitFunction(func::FuncOp func) {
    if (!func.getBody().hasOneBlock())
      return func.emitError("WaveMachine AMDGPU emitter supports one-block funcs");

    os << "\n\t.globl\t" << func.getSymName() << "\n";
    os << "\t.p2align\t8\n";
    os << "\t.type\t" << func.getSymName() << ",@function\n";
    os << func.getSymName() << ":\n";
    emitLine(StringRef("; wave backend: WaveMachine MLIR pipeline finalized"));

    for (Operation &op : func.getBody().front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      if (!isWM(&op))
        return op.emitError("unexpected non-WaveMachine operation in emitter");
      if (failed(emitOperation(op)))
        return failure();
    }

    os << "\t.size\t" << func.getSymName() << ", .-" << func.getSymName()
       << "\n";
    if (func->hasAttr("wave.kernel")) {
      KernelInfo info;
      info.name = func.getSymName().str();
      info.kernargSize = getKernelArgSize(func);
      info.sgprCount = getIntAttr(func, "wavemachine.sgpr_count", 6);
      info.vgprCount = getIntAttr(func, "wavemachine.vgpr_count", 1);
      unsigned offset = 0;
      for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
        bool isBuffer = isa<wave::PtrType>(arg.getType());
        unsigned size = kernelArgSize(arg.getType());
        info.args.push_back(KernelArgInfo{("arg" + Twine(index)).str(), offset,
                                          size, isBuffer && !isBufferPointer(arg.getType())});
        offset += size;
      }
      kernels.push_back(info);
      emitKernelDescriptor(func);
    }
    return success();
  }

  unsigned getKernelArgSize(func::FuncOp func) const {
    if (auto attr = func->getAttrOfType<IntegerAttr>("wavemachine.kernarg_size"))
      return attr.getInt();
    unsigned size = 0;
    for (BlockArgument arg : func.getArguments())
      size += kernelArgSize(arg.getType());
    return (std::max(size, 4u) + 7u) & ~7u;
  }

  void emitKernelDescriptor(func::FuncOp func) {
    unsigned kernargSize = getKernelArgSize(func);
    unsigned sgprCount = getIntAttr(func, "wavemachine.sgpr_count", 6);
    unsigned vgprCount = getIntAttr(func, "wavemachine.vgpr_count", 1);
    os << "\t.section\t.rodata,\"a\",@progbits\n";
    os << "\t.p2align\t6, 0x0\n";
    os << "\t.amdhsa_kernel " << func.getSymName() << "\n";
    os << "\t\t.amdhsa_group_segment_fixed_size 0\n";
    os << "\t\t.amdhsa_private_segment_fixed_size 0\n";
    os << "\t\t.amdhsa_kernarg_size " << kernargSize << "\n";
    os << "\t\t.amdhsa_user_sgpr_count 2\n";
    os << "\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1\n";
    os << "\t\t.amdhsa_wavefront_size32 1\n";
    os << "\t\t.amdhsa_uses_dynamic_stack 0\n";
    os << "\t\t.amdhsa_enable_private_segment 0\n";
    os << "\t\t.amdhsa_system_sgpr_workgroup_id_x 1\n";
    os << "\t\t.amdhsa_system_sgpr_workgroup_id_y 0\n";
    os << "\t\t.amdhsa_system_sgpr_workgroup_id_z 0\n";
    os << "\t\t.amdhsa_system_sgpr_workgroup_info 0\n";
    os << "\t\t.amdhsa_system_vgpr_workitem_id 0\n";
    os << "\t\t.amdhsa_next_free_vgpr " << vgprCount << "\n";
    os << "\t\t.amdhsa_next_free_sgpr " << sgprCount << "\n";
    os << "\t\t.amdhsa_reserve_vcc 0\n";
    os << "\t\t.amdhsa_float_round_mode_32 0\n";
    os << "\t\t.amdhsa_float_round_mode_16_64 0\n";
    os << "\t\t.amdhsa_float_denorm_mode_32 3\n";
    os << "\t\t.amdhsa_float_denorm_mode_16_64 3\n";
    os << "\t\t.amdhsa_dx10_clamp 1\n";
    os << "\t\t.amdhsa_ieee_mode 1\n";
    os << "\t\t.amdhsa_fp16_overflow 0\n";
    os << "\t\t.amdhsa_workgroup_processor_mode 1\n";
    os << "\t\t.amdhsa_memory_ordered 1\n";
    os << "\t\t.amdhsa_forward_progress 1\n";
    os << "\t\t.amdhsa_shared_vgpr_count 0\n";
    os << "\t\t.amdhsa_inst_pref_size 1\n";
    os << "\t.end_amdhsa_kernel\n";
    os << "\t.text\n";
    os << "\t.set .L" << func.getSymName() << ".num_vgpr, " << vgprCount
       << "\n";
    os << "\t.set .L" << func.getSymName() << ".num_agpr, 0\n";
    os << "\t.set .L" << func.getSymName() << ".numbered_sgpr, " << sgprCount
       << "\n";
    os << "\t.set .L" << func.getSymName() << ".num_named_barrier, 0\n";
    os << "\t.set .L" << func.getSymName() << ".private_seg_size, 0\n";
    os << "\t.set .L" << func.getSymName() << ".uses_vcc, 0\n";
    os << "\t.set .L" << func.getSymName() << ".uses_flat_scratch, 0\n";
    os << "\t.set .L" << func.getSymName() << ".has_dyn_sized_stack, 0\n";
    os << "\t.set .L" << func.getSymName() << ".has_recursion, 0\n";
    os << "\t.set .L" << func.getSymName() << ".has_indirect_call, 0\n";
  }

  void emitMetadata() {
    if (kernels.empty())
      return;

    os << "\t.amdgpu_metadata\n";
    os << "---\n";
    os << "amdhsa.kernels:\n";
    for (const KernelInfo &kernel : kernels) {
      os << "  - .args:\n";
      for (const KernelArgInfo &arg : kernel.args) {
        if (arg.isGlobalBuffer) {
          os << "      - .address_space:  global\n";
          os << "        .name:           " << arg.name << "\n";
          os << "        .offset:         " << arg.offset << "\n";
          os << "        .size:           " << arg.size << "\n";
          os << "        .value_kind:     global_buffer\n";
        } else {
          os << "      - .name:           " << arg.name << "\n";
          os << "        .offset:         " << arg.offset << "\n";
          os << "        .size:           " << arg.size << "\n";
          os << "        .value_kind:     by_value\n";
        }
      }
      os << "    .group_segment_fixed_size: 0\n";
      os << "    .kernarg_segment_align: 8\n";
      os << "    .kernarg_segment_size: " << kernel.kernargSize << "\n";
      os << "    .max_flat_workgroup_size: 1024\n";
      os << "    .name:           " << kernel.name << "\n";
      os << "    .private_segment_fixed_size: 0\n";
      os << "    .sgpr_count:     " << kernel.sgprCount << "\n";
      os << "    .sgpr_spill_count: 0\n";
      os << "    .symbol:         " << kernel.name << ".kd\n";
      os << "    .uses_dynamic_stack: false\n";
      os << "    .vgpr_count:     " << kernel.vgprCount << "\n";
      os << "    .vgpr_spill_count: 0\n";
      os << "    .wavefront_size: 32\n";
      os << "    .workgroup_processor_mode: 1\n";
    }
    os << "amdhsa.target:   amdgcn-amd-amdhsa--gfx1100\n";
    os << "amdhsa.version:\n";
    os << "  - 1\n";
    os << "  - 2\n";
    os << "...\n";
    os << "\t.end_amdgpu_metadata\n";
  }

  unsigned getPhys(Value value) const {
    auto regType = cast<wavemachine::RegType>(value.getType());
    if (regType.getIndex() >= 0)
      return regType.getIndex();
    llvm_unreachable("expected allocated WaveMachine register");
  }

  std::string physReg(Value value) const {
    auto regType = cast<wavemachine::RegType>(value.getType());
    unsigned phys = getPhys(value);
    StringRef prefix =
        regType.getRegClass() == wavemachine::RegClass::VGPR ? "v" : "s";
    if (regType.getWidth() == 1)
      return (prefix + Twine(phys)).str();
    return (prefix + Twine("[") + Twine(phys) + ":" +
            Twine(phys + regType.getWidth() - 1) + "]")
        .str();
  }

  std::string operandToString(Value value) const {
    Operation *def = value.getDefiningOp();
    if (isa<wavemachine::ImmOp>(def))
      return Twine(def->getAttrOfType<IntegerAttr>("value").getInt()).str();
    return physReg(value);
  }

  unsigned namedPhysReg(StringRef name) const {
    if (name == "s0")
      return llvm::AMDGPU::SGPR0;
    if (name == "s[0:1]")
      return llvm::AMDGPU::SGPR0_SGPR1;
    if (name == "exec_lo")
      return llvm::AMDGPU::EXEC_LO;
    if (name == "null")
      return llvm::AMDGPU::SGPR_NULL;
    llvm_unreachable("unknown physical register name");
  }

  unsigned mcReg(Value value) const {
    auto regType = cast<wavemachine::RegType>(value.getType());
    unsigned phys = getPhys(value);
    if (regType.getRegClass() == wavemachine::RegClass::VGPR)
      return mcVGPRReg(phys, regType.getWidth());
    if (regType.getWidth() == 4)
      return llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3 + phys / 4;
    if (regType.getWidth() == 2)
      return llvm::AMDGPU::SGPR0_SGPR1 + phys / 2;
    return llvm::AMDGPU::SGPR0 + phys;
  }

  unsigned mcVGPRReg(unsigned phys, unsigned width) const {
    switch (width) {
    case 1:
      return llvm::AMDGPU::VGPR0 + phys;
    case 2:
      return llvm::AMDGPU::VGPR0_VGPR1 + phys;
    case 4:
      return llvm::AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3 + phys;
    case 8:
      return llvm::AMDGPU::
                 VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7 +
             phys;
    default:
      llvm_unreachable("unsupported VGPR tuple width");
    }
  }

  llvm::MCOperand toMCVGPRComponent(Value value, unsigned component) const {
    auto regType = cast<wavemachine::RegType>(value.getType());
    if (regType.getRegClass() != wavemachine::RegClass::VGPR ||
        component >= regType.getWidth())
      llvm_unreachable("expected valid VGPR tuple component");
    return llvm::MCOperand::createReg(mcVGPRReg(getPhys(value) + component, 1));
  }

  llvm::MCOperand toMCSGPRComponent(Value value, unsigned component) const {
    auto regType = cast<wavemachine::RegType>(value.getType());
    if (regType.getRegClass() != wavemachine::RegClass::SGPR ||
        component >= regType.getWidth())
      llvm_unreachable("expected valid SGPR tuple component");
    return llvm::MCOperand::createReg(llvm::AMDGPU::SGPR0 + getPhys(value) +
                                     component);
  }

  llvm::MCOperand toMCOperand(Value value) {
    Operation *def = value.getDefiningOp();
    if (isa<wavemachine::ImmOp>(def))
      return llvm::MCOperand::createImm(
          def->getAttrOfType<IntegerAttr>("value").getInt());
    return llvm::MCOperand::createReg(mcReg(value));
  }

  llvm::MCOperand labelOperand(StringRef name) {
    llvm::MCSymbol *sym = mcContext->getOrCreateSymbol(name);
    return llvm::MCOperand::createExpr(
        llvm::MCSymbolRefExpr::create(sym, *mcContext));
  }

  LogicalResult emitMC(unsigned opcode, ArrayRef<llvm::MCOperand> operands) {
    llvm::MCInst inst;
    inst.setOpcode(opcode);
    for (const llvm::MCOperand &operand : operands)
      inst.addOperand(operand);
    for (unsigned i = 0; i < indent; ++i)
      os << '\t';
    instPrinter->printInst(&inst, /*Address=*/0, /*Annot=*/"", *sti, os);
    os << '\n';
    return success();
  }

  LogicalResult emitMCValues(unsigned opcode, ValueRange operands) {
    SmallVector<llvm::MCOperand> mcOperands;
    for (Value operand : operands)
      mcOperands.push_back(toMCOperand(operand));
    return emitMC(opcode, mcOperands);
  }

  bool isSGPR(Value value) const {
    auto regType = dyn_cast<wavemachine::RegType>(value.getType());
    return regType && regType.getRegClass() == wavemachine::RegClass::SGPR;
  }

  LogicalResult emitOperation(Operation &op) {
    auto operandString = [&](unsigned i) { return operandToString(op.getOperand(i)); };
    auto result = [&]() { return op.getResult(0); };
    StringRef name = op.getName().getStringRef();

    if (isa<wavemachine::ImmOp, wavemachine::ArgOp, wavemachine::TokenOp,
            wavemachine::TokenJoinOp, wavemachine::WaitOp>(&op))
      return success();
    if (isa<wavemachine::LabelOp>(op)) {
      os << op.getAttrOfType<StringAttr>("name").str() << ":\n";
      return success();
    }
    if (isa<wavemachine::VMbcntLoOp>(op))
      return emitMC(llvm::AMDGPU::V_MBCNT_LO_U32_B32_e64_gfx11,
                    {toMCOperand(result()), llvm::MCOperand::createImm(-1),
                     llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::VMovB32TupleOp>(op)) {
      auto regType = cast<wavemachine::RegType>(result().getType());
      for (unsigned i = 0, e = regType.getWidth(); i != e; ++i)
        if (failed(emitMC(llvm::AMDGPU::V_MOV_B32_e32_gfx11,
                          {toMCVGPRComponent(result(), i),
                           toMCOperand(op.getOperand(0))})))
          return failure();
      return success();
    }
    if (isa<wavemachine::WmmaI32_16x16x16_IU8Op>(op))
      return emitMC(llvm::AMDGPU::V_WMMA_I32_16X16X16_IU8_twoaddr_w32_gfx11,
                    {toMCOperand(result()),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(0)),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(1)),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(2)),
                     llvm::MCOperand::createImm(0),
                     llvm::MCOperand::createImm(0),
                     llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::WmmaF32_16x16x16_F16Op>(op))
      return emitMC(llvm::AMDGPU::V_WMMA_F32_16X16X16_F16_twoaddr_w32_gfx11,
                    {toMCOperand(result()),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(0)),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(1)),
                     llvm::MCOperand::createImm(0),
                     toMCOperand(op.getOperand(2)),
                     llvm::MCOperand::createImm(0),
                     llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::VAddU32Op>(op)) {
      Value lhs = op.getOperand(0);
      Value rhs = op.getOperand(1);
      if (isSGPR(rhs))
        std::swap(lhs, rhs);
      return emitMC(llvm::AMDGPU::V_ADD_NC_U32_e32_gfx11,
                    {toMCOperand(result()), toMCOperand(lhs), toMCOperand(rhs)});
    }
    if (isa<wavemachine::VAndB32Op, wavemachine::VOrB32Op,
            wavemachine::VXorB32Op>(op)) {
      Value lhs = op.getOperand(0);
      Value rhs = op.getOperand(1);
      if (isSGPR(rhs))
        std::swap(lhs, rhs);
      unsigned opcode =
          isa<wavemachine::VAndB32Op>(op) ? llvm::AMDGPU::V_AND_B32_e32_gfx11
          : isa<wavemachine::VOrB32Op>(op) ? llvm::AMDGPU::V_OR_B32_e32_gfx11
                                           : llvm::AMDGPU::V_XOR_B32_e32_gfx11;
      return emitMC(opcode,
                    {toMCOperand(result()), toMCOperand(lhs), toMCOperand(rhs)});
    }
    if (isa<wavemachine::VLshlrevB32Op>(op))
      return emitMC(llvm::AMDGPU::V_LSHLREV_B32_e32_gfx11,
                    {toMCOperand(result()), toMCOperand(op.getOperand(1)),
                     toMCOperand(op.getOperand(0))});
    if (isa<wavemachine::VCmpEqU32Op, wavemachine::VCmpNeU32Op,
            wavemachine::VCmpLtU32Op, wavemachine::VCmpLeU32Op,
            wavemachine::VCmpGtU32Op, wavemachine::VCmpGeU32Op>(op)) {
      unsigned opcode =
          isa<wavemachine::VCmpEqU32Op>(op) ? llvm::AMDGPU::V_CMP_EQ_U32_e64_gfx11
          : isa<wavemachine::VCmpNeU32Op>(op) ? llvm::AMDGPU::V_CMP_NE_U32_e64_gfx11
          : isa<wavemachine::VCmpLtU32Op>(op) ? llvm::AMDGPU::V_CMP_LT_U32_e64_gfx11
          : isa<wavemachine::VCmpLeU32Op>(op) ? llvm::AMDGPU::V_CMP_LE_U32_e64_gfx11
          : isa<wavemachine::VCmpGtU32Op>(op) ? llvm::AMDGPU::V_CMP_GT_U32_e64_gfx11
                                               : llvm::AMDGPU::V_CMP_GE_U32_e64_gfx11;
      return emitMC(opcode,
                    {toMCOperand(result()), toMCOperand(op.getOperand(0)),
                     toMCOperand(op.getOperand(1))});
    }
    if (isa<wavemachine::SMovB32Op>(op)) {
      StringRef dst = op.getAttrOfType<StringAttr>("dst").getValue();
      std::string src = operandString(0);
      if (dst != src)
        emitLine(Twine("s_mov_b32 ") + dst + ", " + src);
      return success();
    }
    if (isa<wavemachine::SLoadB32Op>(op))
      return emitMC(llvm::AMDGPU::S_LOAD_B32_IMM_gfx11,
                    {toMCOperand(result()),
                     llvm::MCOperand::createReg(
                         namedPhysReg(op.getAttrOfType<StringAttr>("base").getValue())),
                     toMCOperand(op.getOperand(0)), llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::SLoadB64Op>(op))
      return emitMC(llvm::AMDGPU::S_LOAD_B64_IMM_gfx11,
                    {toMCOperand(result()),
                     llvm::MCOperand::createReg(
                         namedPhysReg(op.getAttrOfType<StringAttr>("base").getValue())),
                     toMCOperand(op.getOperand(0)), llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::SLoadB128Op>(op))
      return emitMC(llvm::AMDGPU::S_LOAD_B128_IMM_gfx11,
                    {toMCOperand(result()),
                     llvm::MCOperand::createReg(
                         namedPhysReg(op.getAttrOfType<StringAttr>("base").getValue())),
                     toMCOperand(op.getOperand(0)), llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::SWaitcntOp>(op))
      return emitMCValues(llvm::AMDGPU::S_WAITCNT_gfx11, op.getOperands());
    if (isa<wavemachine::SWaitcntVscntOp>(op))
      return emitMC(llvm::AMDGPU::S_WAITCNT_VSCNT_gfx11,
                    {llvm::MCOperand::createReg(namedPhysReg("null")),
                     toMCOperand(op.getOperand(0))});
    if (isa<wavemachine::SNopOp>(op))
      return emitMCValues(llvm::AMDGPU::S_NOP_gfx11, op.getOperands());
    if (isa<wavemachine::SDelayAluOp>(op))
      return emitMCValues(llvm::AMDGPU::S_DELAY_ALU_gfx11, op.getOperands());
    if (isa<wavemachine::SAndSaveexecB32Op>(op))
      return emitMC(llvm::AMDGPU::S_AND_SAVEEXEC_B32_gfx11,
                    {toMCOperand(result()), toMCOperand(op.getOperand(0))});
    if (isa<wavemachine::SAndn2ExecB32Op>(op)) {
      emitLine(Twine("s_andn2_b32 exec_lo, ") + operandString(0) + ", " +
               operandString(1));
      return success();
    }
    if (isa<wavemachine::SCBranchExeczOp>(op))
      return emitMC(llvm::AMDGPU::S_CBRANCH_EXECZ_gfx11,
                    {labelOperand(op.getAttrOfType<StringAttr>("label"))});
    if (isa<wavemachine::SMovExecLoOp>(op)) {
      emitLine(Twine("s_mov_b32 exec_lo, ") + operandString(0));
      return success();
    }
    if (isa<wavemachine::VReadfirstlaneB32Op>(op))
      return emitMC(llvm::AMDGPU::V_READFIRSTLANE_B32_gfx11,
                    {toMCOperand(result()), toMCOperand(op.getOperand(0))});
    if (isa<wavemachine::GlobalStoreB32Op>(op))
      return emitMC(llvm::AMDGPU::GLOBAL_STORE_DWORD_SADDR_gfx11,
                    {toMCOperand(op.getOperand(0)), toMCOperand(op.getOperand(1)),
                     toMCOperand(op.getOperand(2)), llvm::MCOperand::createImm(0),
                     llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::MakeBufferRsrcOp>(op)) {
      constexpr uint32_t gfx11Format32Float = 22;
      constexpr uint32_t defaultRsrcFlags =
          (gfx11Format32Float << 12) | (1u << 24) | (3u << 28);
      if (failed(emitMC(llvm::AMDGPU::S_MOV_B32_gfx11,
                        {toMCSGPRComponent(result(), 0),
                         toMCSGPRComponent(op.getOperand(0), 0)})) ||
          failed(emitMC(llvm::AMDGPU::S_MOV_B32_gfx11,
                        {toMCSGPRComponent(result(), 1),
                         toMCSGPRComponent(op.getOperand(0), 1)})) ||
          failed(emitMC(llvm::AMDGPU::S_MOV_B32_gfx11,
                        {toMCSGPRComponent(result(), 2),
                         toMCOperand(op.getOperand(1))})) ||
          failed(emitMC(llvm::AMDGPU::S_MOV_B32_gfx11,
                        {toMCSGPRComponent(result(), 3),
                         llvm::MCOperand::createImm(defaultRsrcFlags)})))
        return failure();
      return success();
    }
    if (isa<wavemachine::BufferStoreB32Op>(op)) {
      emitLine(Twine("buffer_store_dword ") + operandString(1) + ", " +
               operandString(0) + ", " + physReg(op.getOperand(2)) +
               ", 0 offen");
      return success();
    }
    if (isa<wavemachine::GlobalStoreTupleB32Op>(op)) {
      unsigned component = getIntAttr(&op, "component", 0);
      return emitMC(llvm::AMDGPU::GLOBAL_STORE_DWORD_SADDR_gfx11,
                    {toMCOperand(op.getOperand(0)),
                     toMCVGPRComponent(op.getOperand(1), component),
                     toMCOperand(op.getOperand(2)),
                     llvm::MCOperand::createImm(component * 4),
                     llvm::MCOperand::createImm(0)});
    }
    if (isa<wavemachine::SEndpgmOp>(op))
      return emitMC(llvm::AMDGPU::S_ENDPGM_gfx11, {llvm::MCOperand::createImm(0)});
    if (isa<wavemachine::SSetpcB64Op>(op)) {
      emitLine(StringRef("s_setpc_b64 s[30:31]"));
      return success();
    }

    return op.emitError("unsupported WaveMachine opcode: ") << name;
  }
};

static LogicalResult runWaveMachinePipeline(ModuleOp module) {
  Builder builder(module.getContext());
  module->setAttr("wavemachine.target",
                  builder.getStringAttr("amdgcn-amd-amdhsa--gfx1100"));
  PassManager pm(module.getContext());
  pm.addPass(wave::createConvertWaveAMDToWaveMachine());
  pm.addPass(wave::createWaveAMDABILowering());
  pm.addPass(wave::createWaveAMDTicketWaits());
  pm.addPass(wave::createWaveAMDHazardWaits());
  pm.addPass(wave::createWaveAMDRegAlloc());
  pm.addPass(wave::createWaveAMDResourceInfo());
  pm.addPass(wave::createWaveAMDMetadata());
  return pm.run(module);
}

} // namespace

LogicalResult mlir::wave::translateWaveToAMDGPU(Operation *op,
                                                raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("wave AMDGPU backend expects a module operation");
  if (failed(runWaveMachinePipeline(module)))
    return failure();
  return WaveAMDGPUEmitter(os).emit(module);
}
